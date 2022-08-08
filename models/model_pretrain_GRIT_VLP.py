from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM

import torch
import torch.nn.functional as F
from torch import nn

class ALBEF_Base(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.itm_head = nn.Linear(text_width, 2)     
        


    def forward(self, image, text): 
        
        with torch.no_grad():
            bs=image.size(0)
            self.temp.clamp_(0.001,0.5)
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  
        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 
        
        with torch.no_grad():
            image_feat_sg = image_feat.t().clone()
            text_feat_sg = text_feat.t().clone()   
        
        sim_i2t = image_feat @ text_feat_sg / self.temp 
        sim_t2i = text_feat @ image_feat_sg / self.temp 
        with torch.no_grad():
                sim_targets = torch.zeros(sim_i2t.size()).to(image.device)
                sim_targets.fill_diagonal_(1)
                image_feat_store=image_feat.clone().detach()
                text_feat_store=text_feat.clone().detach()
                
                sim_i2t_sg= sim_i2t.clone().detach()
                sim_t2i_sg= sim_t2i.clone().detach()

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

        loss_c_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)* F.softmax(sim_t2i_sg, dim=1),  dim=1).mean()
        loss_c_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)* F.softmax(sim_i2t_sg, dim=1),  dim=1).mean()
        
        loss_ita = (loss_i2t+loss_t2i)/2 + 0.2* (loss_c_i2t+loss_c_t2i)/2
        
        ###=================================###
        # forward the positve image-text pair
        output_pos = self.text_encoder.bert(encoder_embeds = text_embeds,
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,
                                        return_dict = True,
                                        mode = 'fusion',
                                       )

        with torch.no_grad():
            bs = image.size(0)
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        image_neg_idx=[]
        for b in range(bs):
            top_k= torch.topk(weights_t2i[b],1)
            neg_idx= top_k.indices[0] 
            image_embeds_neg.append(image_embeds[neg_idx])
            image_neg_idx.append(neg_idx)
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        text_neg_idx=[]
        for b in range(bs):
            top_k= torch.topk(weights_i2t[b],1)
            neg_idx= top_k.indices[0]          
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
            text_neg_idx.append(neg_idx)
        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)
        text_atts_neg = torch.stack(text_atts_neg,dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder.bert(encoder_embeds = text_embeds_all,
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,
                                        return_dict = True,
                                        mode = 'fusion',
                                       )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)


        ##================= MLM ========================##                
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        
        mlm_output = self.text_encoder(input_ids, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       labels = labels,   
                                      )                           
        loss_mlm = mlm_output.loss        

        return loss_mlm, loss_ita, loss_itm, image_feat_store, text_feat_store
        
        
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

