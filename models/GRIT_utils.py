import torch
import torch.nn.functional as F
import random
import numpy as np

class GRIT():
    def __init__(self,
                 config,device,num_steps):

        batch_size=config['batch_size']
        embed_dim = config['embed_dim']

        self.search_space=config['search_space']
        total_sample_len= num_steps *batch_size
        self.G_index_set= list(range(total_sample_len))
        self.G_idx=0
        self.queue_size= config['queue_size']
        self.num_small_queue  = int(config['queue_size']/self.search_space)
        
        # For simplicity
        assert self.queue_size % self.search_space == 0  
        assert self.queue_size % batch_size == 0  
        assert self.search_space % batch_size ==0 

        self.image_queue= torch.randn(embed_dim, self.queue_size).to(device)
        self.text_queue= torch.randn(embed_dim, self.queue_size).to(device)
        self.idx_queue= torch.randn(self.queue_size).to(device)
        self.queue_ptr= torch.zeros(1, dtype=torch.long)

    @torch.no_grad()
    def grit_second_third_phase(self,cur_step,temp,num_steps):
        if self.queue_ptr[0]  == 0:
            self.example_level_shuffle()

            # Divide / Grouping 
            for q in range(self.num_small_queue):
                # Grouping
                index_m = self.grouping(self.image_queue[:,q*self.search_space:(q+1)*self.search_space ],self.text_queue[:,q*self.search_space:(q+1)*self.search_space ],self.idx_queue[q*self.search_space:(q+1)*self.search_space ],temp)
                
                # fill in G
                self.G_index_set[(self.G_idx)*self.search_space:(self.G_idx+1)*self.search_space] = index_m
                self.G_idx+=1

        elif cur_step== num_steps-1:
            # Example level Shuffle           
            self.remaining_queue()
            self.example_level_shuffle()
            slice_queue= int (int (self.queue_ptr[0]) // self.search_space )

            if slice_queue >0:
                for q in range(slice_queue):
                    index_m = self.grouping(self.image_queue[:,q*self.search_space:(q+1)*self.search_space],self.text_queue[:,q*self.search_space:(q+1)*self.search_space],self.idx_queue[q*self.search_space:(q+1)*self.search_space],temp)
                
                    self.G_index_set[(self.G_idx)*self.search_space:(self.G_idx+1)*self.search_space] = index_m
                    self.G_idx+=1

            # Remaining indices
            index_m = self.grouping(self.image_queue[:,slice_queue*self.search_space:self.queue_ptr[0]],self.text_queue[:,slice_queue*self.search_space:self.queue_ptr[0]],self.idx_queue[slice_queue*self.search_space:self.queue_ptr[0]],temp)
            self.G_index_set[(self.G_idx)*self.search_space:] = index_m
            self.G_idx+=1



    @torch.no_grad()
    def grouping(self,image_sub_queue,text_sub_queue,index_sub_queue,temp):
            sim_i2t_sg =  F.softmax(image_sub_queue.detach().t() @ text_sub_queue.detach() / temp,dim=1)
            sim_i2t_sg.fill_diagonal_(0)
            sim_t2i_sg =  F.softmax(text_sub_queue.detach().t() @ image_sub_queue.detach() / temp,dim=1)
            sim_t2i_sg.fill_diagonal_(0)

            bs= image_sub_queue.size()[1]
            I_index_set=[]
            start = torch.randint(low=0, high=int(bs-1),size=(1,))[0]
            start = start.to(image_sub_queue.device)
            next_i_idx=start
            I_index_set.append(index_sub_queue[start].to(torch.long).detach().cpu())
            
            group_iter=int((bs-1)//2)

            for group_idx in range(group_iter):
                next_t=  torch.topk(sim_i2t_sg[next_i_idx],1)
                next_t_idx=next_t.indices[0]
                sim_i2t_sg[next_i_idx,:]=0
                sim_i2t_sg[:,next_i_idx]=0
                sim_t2i_sg[next_i_idx,:]=0
                sim_t2i_sg[:,next_i_idx]=0
                next_i = torch.topk(sim_t2i_sg[next_t_idx],1)
                next_i_idx=next_i.indices[0]

                sim_i2t_sg[next_t_idx,:]=0
                sim_i2t_sg[:,next_t_idx]=0
                sim_t2i_sg[next_t_idx,:]=0
                sim_t2i_sg[:,next_t_idx]=0
                I_index_set.append(index_sub_queue[next_t_idx].to(torch.long).detach().cpu())
                I_index_set.append(index_sub_queue[next_i_idx].to(torch.long).detach().cpu())

            if int((bs-1)%2) !=0:
                next_t_idx=  torch.argmax(sim_i2t_sg[next_i_idx,:])
                I_index_set.append(index_sub_queue[next_t_idx].to(torch.long).detach().cpu())

            return I_index_set   


    @torch.no_grad()
    def example_level_shuffle(self):
        shuffle_idx = torch.randperm(self.image_queue.shape[1])
        self.image_queue=self.image_queue[:,shuffle_idx].view(self.image_queue.size())
        self.text_queue=self.text_queue[:,shuffle_idx].view(self.text_queue.size())
        self.idx_queue=self.idx_queue[shuffle_idx].view(self.idx_queue.size())

    @torch.no_grad()
    def remaining_queue (self):
        self.image_queue = self.image_queue[:,:self.queue_ptr[0]]
        self.text_queue = self.text_queue[:,:self.queue_ptr[0]]
        self.idx_queue = self.idx_queue[:self.queue_ptr[0]]



    @torch.no_grad()
    def collecting(self, image_feat, text_feat,idx):
        batch_size = image_feat.shape[0]
        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feat.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feat.T
        self.idx_queue[ ptr:ptr + batch_size] = idx.detach()
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


@torch.no_grad()
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
            yield lst[i:i + n]


@torch.no_grad()
def mini_batch_level_shuffle(index_set, batch_size):
    divided_G_index_set = list(chunks(index_set,batch_size))
    total_chunk_size = len(divided_G_index_set)
    chunk_arr = np.arange(total_chunk_size)
    random.shuffle(chunk_arr)
    shuffled_G_index_set=[]
    for ind in chunk_arr:
        shuffled_G_index_set += divided_G_index_set[ind]

    return shuffled_G_index_set

