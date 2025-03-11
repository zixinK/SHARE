# from memory_profiler import profile
import os
import psutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence
from modeling.metric import AUPRC
import Levenshtein
from modeling.MLP import MLPRegressor,mlp_train
from modeling.Rationale_editor import continuity,rl_cd_ss,token_dis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers.file_utils import ModelOutput
from transformers import (
    T5ForConditionalGeneration,
    T5EncoderModel,
    T5PreTrainedModel,
    AutoTokenizer
    )

from transformers.utils import logging
logging.set_verbosity_warning()

from dataclasses import dataclass
from typing import Optional, Tuple

import warnings
warnings.filterwarnings("ignore")


# from modeling.sigmoid import sigmoid_function

from modeling.connected_shapley import explain_shapley, construct_positions_connectedshapley
# from modeling.continuity import  continuity,rl_cd_ss
from modeling.logical_fluency import newset

# @profile
def compute_metrics_fn(eval_prediction, tokenizer=None, return_text=False, nei_index=2):

    loss = eval_prediction[0]
    
    s_loss = eval_prediction[6]
    cph_loss = eval_prediction[7]
    cpt_loss = eval_prediction[8]
    lf_loss = eval_prediction[9]
    lf_loss_gt = eval_prediction[10]
    evidence_len = eval_prediction[11]
    token_dis = eval_prediction[12]
    # full_preds = np.argmax(full_logits, axis=1)
    
    # mlp_loss = eval_prediction[8]
    # lf_loss = eval_prediction[9]
    # lf_loss_gt = eval_prediction[10]
    # evidence_len = eval_prediction[11]
    
    acc=[]
    f1 = []
    recall = []
    precision = []
    ed = []
    ed_p = []
    auprc = []
    select_r = []
    ground_r = []
    lf_acc = []
    nolf_pred=[]
    lf_pred = []
    for i in range(len(eval_prediction[1])):
        masked_logits = eval_prediction[1][i]
        mask = eval_prediction[2][i]
        labels = eval_prediction[3][i]
        mask_labels = eval_prediction[4][i]
        logical_fluency_logits =  eval_prediction[13][i]
        mask_labels = np.maximum(mask_labels, 0)
        masked_preds = np.argmax(masked_logits, axis=1).tolist()
        # logical_gluency_preds = np.argmax(logical_fluency_logits, axis=1).tolist()
        # masked_preds = [[m_p] for m_p in masked_preds]
        is_masked = np.greater(mask, 0.5).astype(int).tolist()
        if isinstance(masked_logits[0],list):
            masked_preds = np.argmax(masked_logits, axis=1).tolist()
        else:
            labels = [labels]
            masked_preds = [np.argmax(masked_logits, axis=0).tolist()]
            mask_labels = [mask_labels]
            is_masked = [is_masked]
        # print(masked_preds)
        # print(labels)
        acc.append(metrics.accuracy_score(labels, masked_preds))
        nolf_pred.append(masked_logits)
        lf_pred.append(logical_fluency_logits)
        # lf_acc.append(metrics.accuracy_score(labels, logical_gluency_preds))
        # f1.append(metrics.f1_score(is_masked,mask_labels, average="micro"))
        # recall.append(metrics.recall_score(is_masked,mask_labels, average="micro"))
        # precision.append(metrics.precision_score(is_masked,mask_labels, average="micro"))
        for j in range(len(mask_labels)):
            # f1.append(metrics.f1_score(mask_labels, is_masked, average="micro", zero_division=1))
            # print(np.sum(is_masked[j]))
            # print(is_masked[j])
            # print(np.sum(mask_labels[j]))
            # print(mask_labels[j].astype(np.int).tolist())
            # print('----------------------------------------------')
            # print(sum(is_masked[j]))
            # print(sum(mask_labels[j]))
            select_r.append(is_masked[j])
            ground_r.append(mask_labels[j].astype(int).tolist())
            f1.append(metrics.f1_score(is_masked[j],mask_labels[j].astype(int).tolist(), average="micro"))
            recall.append(metrics.recall_score(is_masked[j],mask_labels[j].astype(int).tolist(), average="micro"))
            precision.append(metrics.precision_score(is_masked[j],mask_labels[j].astype(int).tolist(), average="micro"))
            # f1.append(metrics.f1_score(is_masked[j],mask_labels[j].astype(np.int).tolist(), average="micro", zero_division=1))
            # recall.append(metrics.recall_score(is_masked[j],mask_labels[j].astype(np.int).tolist(), average="micro", zero_division=1))
            # precision.append(metrics.precision_score(is_masked[j],mask_labels[j].astype(np.int).tolist(), average="micro", zero_division=1))
            
        # for j in range(len(mask_labels)):
            e_d= Levenshtein.distance(is_masked[j],mask_labels[j].astype(int).tolist())
            ed.append(e_d)
            ed_p.append(e_d/(evidence_len[i][j]))
        auprc.append(np.mean(AUPRC(mask_labels, is_masked)))
    
    # np.save('movie_test_albert_rationale_90.npy', select_r)
    # np.save('movie_test_albert_ground_ratinoale_85.npy', ground_r)

    # np.save('mv_test_t5_gtr_lf_pred.npy', lf_pred)
    # np.save('mv_test_t5_gtr_nolf_pred.npy', nolf_pred)    

    results = {}
    # results["full_acc"] = metrics.accuracy_score(labels, full_preds)
    results['loss'] = np.mean(loss)
    results["masked_acc"] = np.mean(acc)
    results["mask_f1"] =np.mean(f1)
    results["mask_recall"] = np.mean(recall)
    results["mask_precision"] = np.mean(precision)
    results["auprc"] = np.mean(auprc)
    results["editing_distance"] = np.mean(ed)
    results["editing_distance_percentage"] = np.mean(ed_p)
    results["logical_fluence(extracted)"] = np.mean(lf_loss)
    results["logical_fluence(ground-truth)"] = np.mean(lf_loss_gt)
    results["token_dis"] = np.mean(token_dis)
    # results["lf_acc"] = np.mean(lf_acc)       
    results["suffiency_loss"] = np.mean(s_loss)
    results["comprehensiveness_loss"] = np.mean(cph_loss)
    results["compactness_loss"] = np.mean(cpt_loss)
    # results["mlp_loss"] = np.mean(mlp_loss)


    

    if return_text:
        input_ids = eval_prediction[5][0]
        examples = []
        for i in range(len(input_ids)):
            # import pdb; pdb.set_trace()
            row = tokenizer.convert_ids_to_tokens(input_ids[i])
            ground_truth = []
            masked = []
            for j in range(len(row)):
                if row[j] in ["[CLS]", "[SEP]", "<pad>"]:
                    continue
                if mask_labels[i][j]:
                    ground_truth.append("▁<mask>")
                else:
                    ground_truth.append(row[j])
                if is_masked[i][j]:
                    masked.append("▁<mask>")
                else:
                    masked.append(row[j])
            ground_truth = tokenizer.convert_tokens_to_string(ground_truth)
            masked = tokenizer.convert_tokens_to_string(masked)
            combined = f"Ground-truth:{ground_truth} OLD: {masked}"
            combined = str(combined)
            examples.append(combined)
        results["text"] = '\n'.join(examples)

    return results


@dataclass
class MaskOutput(ModelOutput):
    rationale: torch.FloatTensor= None
    noise: torch.FloatTensor= None
    r_ind: torch.FloatTensor= None
    r_wd:torch.FloatTensor= None
    s_score:torch.FloatTensor= None
    n_r_weights:torch.FloatTensor= None
    lf_weights:torch.FloatTensor= None
    s_score:torch.FloatTensor= None
    n_r_weights:torch.FloatTensor= None
    mlp_loss:torch.FloatTensor = None
    lf_weights:torch.FloatTensor= None
    


@dataclass
class ClassifyOutput(ModelOutput):
    logits: torch.FloatTensor= None
    loss: torch.FloatTensor= None

@dataclass
class TokenTaggingRationaleOutput(ModelOutput):
    loss: torch.FloatTensor= None
    # full_logits: torch.FloatTensor
    
    masked_logits: torch.FloatTensor= None
    rationale: torch.FloatTensor= None
    labels: Optional[torch.FloatTensor] = None
    mask_labels: Optional[torch.FloatTensor] = None
    
    
    # is_unsupervised: Optional[torch.FloatTensor] = None
    input_ids: Optional[torch.LongTensor] = None
    s_loss: torch.FloatTensor = None
    cph_loss:torch.FloatTensor = None
    cpt_loss:torch.FloatTensor = None
    mlp_loss:torch.FloatTensor = None
    lf_loss:torch.FloatTensor = None
    lf_loss_gt:  torch.FloatTensor= None
    evidence_len:  torch.FloatTensor= None
    token_dis:torch.FloatTensor= None
    logical_fluency_logits:torch.FloatTensor= None


class T5ForTokenRationale(T5PreTrainedModel):

    def __init__(self, config):
        super(T5ForTokenRationale, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.nei_index = config.nei_index
        self.mask_token_id = config.mask_token_id
        self.t5 = T5EncoderModel(config).to(device)
        for pp in self.t5.parameters():
            pp.requires_grad = True
        self.dropout = nn.Dropout(config.dropout_rate).to(device)
        # nn.Dropout: 为了防止或减轻过拟合而使用的函数
        self.masker = nn.Linear(config.d_model, 2).to(device).to(device)
        # 是用来设置网络中的全连接层的，而在全连接层中的输入与输出都是二维张量，一般形状为[batch_size, size]
        
        self.pro_th =  self.config.pro_th


        # shapley
        
        self.max_order = self.config.max_order
        self.num_neighbors = self.config.num_neighbors
        self.top_percentage = self.config.top_percentage
        self.con_count = self.config.con_count

        self.cph_margin = self.config.cph_margin
        self.lf_seed=self.config.lf_seed
        self.lf_num = self.config.lf_num
        
        self.testing = self.config.testing
        
        self.class_fl = nn.Linear(config.hidden_size, config.num_labels)


        if not self.config.use_pretrained_classifier:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.init_weights()
        else:
            self.init_weights()
            self.classifier = T5ForConditionalGeneration.from_pretrained(config.pretrained_classifier_path).to(device)
            if self.config.fix_pretrained_classifier:
                for p in self.classifier.parameters():
                    p.requires_grad = False
                    
        self.mlp = MLPRegressor().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('modeling/moviet5_c')
    
    # @profile
    def mask(
        self,
        input_ids=None,
        attention_mask=None,
        evidence_mask=None,
        labels=None,
        scores = None,
        mask_labels=None,
    ):
        # print('A：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
        
        d = evidence_mask.sum(dim=1)

        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids
        )
        sequence_output = outputs[0]
        logits = self.masker(self.dropout(sequence_output))
        


        with torch.no_grad():
            scores=torch.transpose(scores,1,2)
            sh_p = F.gumbel_softmax(scores, tau=self.config.temperature)[:,:,0]
            lg_p = F.gumbel_softmax(logits, tau=self.config.temperature)[:,:,1]
            guidance = 0.5*sh_p+0.5*lg_p
            guidance = guidance*evidence_mask


            one_hot = guidance.clone()
            one_hot = torch.where(one_hot > self.pro_th, 1., 0.)

            con_ra,td_ch,oh_change=continuity(guidance.clone(),one_hot.clone(),3,evidence_mask)
            max_ind_r,max_ind_l = rl_cd_ss(con_ra,guidance.clone())
            gather_ind = max_ind_r.unsqueeze(1).repeat(1,one_hot.shape[-1]).unsqueeze(1)
            con_ra_m = torch.gather(con_ra,1,gather_ind).squeeze(1).unsqueeze(0)
            # print(con_ra_m.squeeze(0).shape)
            # print(evidence_mask.sum(dim=1))
            td_ch_m = torch.gather(td_ch,1,gather_ind).squeeze(1).unsqueeze(0)
            oh_change_m = torch.gather(oh_change,1,gather_ind).squeeze(1).unsqueeze(0)
            y_train = oh_change_m.clone().squeeze(0)
            guidance = torch.where(guidance ==0.,torch.tensor(10000.).to(device),guidance)
            y_train = torch.where(y_train ==0.,guidance,y_train)
            y_train = torch.where(y_train==2.,torch.tensor(0.).to(device),y_train)
            


            rind,rtds = token_dis(con_ra_m.clone().squeeze(0))
            rind = rind.tolist()
            r_d =[[rind[i][j] for j in range(len(rind[i])) if rind[i][j]!=-1000] for i in range(len(rind))]

            n_ind = []
            for i in range(len(y_train)):
                nw = (1-y_train[i])
                ind = torch.topk(nw,k=int(len(r_d[i])))[1]
                n_oh = [1 if j in ind else 0 for j in range(len(y_train[i]))]
                n_ind.append(n_oh)
            n_ind = torch.tensor(n_ind).to(device)
            
            y_train = y_train*evidence_mask
        
            new_w = con_ra_m-n_ind

        noise_weights =logits.clone()
        noise_weights[:, :, 1] = noise_weights[:, :, 1]-(new_w)* 1e10-(1-evidence_mask)* 1e10

        logits[:, :, 1]= logits[:, :, 1]+(new_w)* 1e10-(1-evidence_mask)* 1e10


        if self.training:
            rationale_weights = F.gumbel_softmax(logits, tau=self.config.temperature).select(2,1)
            rationale_weights = torch.where(rationale_weights==1., y_train, rationale_weights)
            # rationale_weights = torch.where(rationale_weights==1., torch.tensor(0.9).to(device), rationale_weights)
            noise_weights = F.gumbel_softmax(noise_weights, tau=self.config.temperature).select(2,1)
            # noise_weights = torch.where(noise_weights==1., torch.tensor(0.9).to(device), noise_weights)
            noise_weights = torch.where(noise_weights==1., 1-(y_train), noise_weights)
            mlp_loss = torch.tensor([[0.]]).to(device)
        else:    
            # rationale_weights = (con_ra_m*evidence_mask).squeeze(0)
            rationale_weights = mask_labels
            noise_weights = evidence_mask*n_ind
            mlp_loss = torch.tensor([[0.]]).to(device)


        return MaskOutput(
            rationale=rationale_weights,
            noise=noise_weights,
            r_ind=r_d,
            mlp_loss = mlp_loss,
            r_wd = con_ra_m.squeeze(0),
        )


    # @profile
    def classify(
        self,
        input_ids=None,
        mask_weights=None,
        attention_mask=None,
        evidence_mask = None,
        labels=None,
        labels_encode = None,
        pos_encode = None,
        neg_encode = None,
        # labels_id=None,
    ):
#         def embed(ids):

#             embeds = self.classifier.get_input_embeddings()(ids)
#             return embeds

            # Embed inputs.
        input_embeds = self.classifier.encoder(input_ids = input_ids,attention_mask = attention_mask).last_hidden_state
        # embed(input_ids.to(device))
        

            # print('C：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))


        # Targeted mask.
        mask_embeds = input_embeds.clone().fill_(0).to(device)
        # mask_weights = mask_weights +attention_mask-evidence_mask
        mask_weights = mask_weights+attention_mask-evidence_mask
        mask_weights = mask_weights.unsqueeze(-1).to(device)
        

        # Mix embeddings.
        mix_embeds = (1-mask_weights) * mask_embeds +  mask_weights * input_embeds
        # print(mix_embeds)
            # Run model.
            
        inputs = {'inputs_embeds':mix_embeds,'attention_mask':attention_mask,'labels':labels_encode}
        
        outputs = self.classifier(**inputs)
        
        lgp = torch.prod(torch.nn.functional.softmax(outputs.logits, dim=-1).gather(2,pos_encode.unsqueeze(2)),1)
        lgn = torch.prod(torch.nn.functional.softmax(outputs.logits, dim=-1).gather(2,neg_encode.unsqueeze(2)),1)
        if len(lgp.shape) !=2:
            lgp = lgp.unsqueeze(0)
        if len(lgn.shape) !=2:
            lgn = lgn.unsqueeze(0)

        logit = torch.cat((lgn,lgp),1).to(device)

        loss = outputs.loss

        return ClassifyOutput(
            logits=logit,
            loss=loss,
        )
    # @profile
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        evidence_mask=None,
        labels=None,
        # labels_id = None,
        mask_labels=None,
        scores = None,
        labels_encode = None,
            pos_encode =None,
            neg_encode = None,
        # is_unsupervised=None,
        **kwargs,
    ):
        # print('E：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
        # Encode the inputs and predict token-level masking scores.
        mask_output = self.mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            evidence_mask=evidence_mask,
            labels=None,
            scores=scores,
            mask_labels=mask_labels)
        rationale_weights = mask_output.rationale
        # n_w = torch.where(rationale_weights ==0., torch.tensor(0.).to(device), 1-rationale_weights)
        # noise_weights = evidence_mask-mask_output.rationale if self.training else mask_output.noise
        noise_weights = mask_output.noise
        rationale_index = mask_output.r_ind
        r_wd = mask_output.r_wd


        # Get the output with targeted masking.
        if self.training:
            masked_cls_output = self.classify(
            input_ids=input_ids,
            mask_weights=rationale_weights,
            # attention_mask=att_m,
                attention_mask=attention_mask,
            evidence_mask=evidence_mask,
            labels=labels,
            labels_encode = labels_encode,
            pos_encode = pos_encode,
            neg_encode = neg_encode,
            )
        else:
            masked_cls_output = self.classify(
            input_ids=input_ids,
            mask_weights=rationale_weights,
            attention_mask=attention_mask*(rationale_weights+attention_mask-evidence_mask),
                # attention_mask=attention_mask,
            evidence_mask=evidence_mask,
            labels=labels,
            labels_encode = labels_encode,
            pos_encode = pos_encode,
            neg_encode = neg_encode,
            )
        # Suffiency
        suffiency_loss = masked_cls_output.loss if labels is not None else torch.Tensor([0+1e-12]).to(device)
        

        if self.training:
            noise_cls_output = self.classify(
                input_ids=input_ids,
                mask_weights=noise_weights,
                # attention_mask=att_n,
                attention_mask=attention_mask,
                evidence_mask=evidence_mask,
                labels=labels,
            labels_encode = labels_encode,
            pos_encode = pos_encode,
            neg_encode = neg_encode,
            )
            # print(noise_cls_output)
        else:
            noise_cls_output = self.classify(
                input_ids=input_ids,
                mask_weights=noise_weights,
                attention_mask=attention_mask*(noise_weights+attention_mask-evidence_mask),
                # attention_mask=attention_mask,
                evidence_mask=evidence_mask,
                labels=labels,
            labels_encode = labels_encode,
            pos_encode = pos_encode,
            neg_encode = neg_encode,
            )  
            
        
        # Noise
        noise_loss = noise_cls_output.loss if labels is not None else torch.Tensor([0+1e-12]).to(device)
        
         # Comprehensive
        cph_loss = 1 / (1 + torch.exp(noise_loss-suffiency_loss-self.cph_margin+1e-12)) 
        # cph_loss = 1 / (1 + torch.exp(noise_loss-suffiency_loss)) 
        cph_loss = torch.where(cph_loss >= 0.5, torch.tensor(1.).to(device), cph_loss)
        
        suffiency_loss = 1/(1+torch.exp(-suffiency_loss+1e-12)) 
        
       
        # Compactness
        if self.training:
            # att_m = attention_mask*(rationale_weights+attention_mask-evidence_mask+(self.pro_th-0.5)).round().long()
            att_m = attention_mask*(rationale_weights+attention_mask-evidence_mask).round().long()
            mask_w = att_m.sum(dim=1)
        else:
            mask_w = rationale_weights.sum(dim=1)
        t = evidence_mask.sum(dim=1)
        cpt_loss = (mask_w/(t+1e-12))

        # New logical fluency

        if self.training:
            new_weights = torch.zeros(1,evidence_mask.shape[1]).to(device)
            input_s = torch.zeros(1,evidence_mask.shape[1]).to(device)
            for i in range(len(evidence_mask)):
                new_ind = (torch.ones(evidence_mask.shape)*torch.tensor([i for i in range(int(evidence_mask.shape[1]))])).to(device)
                idx= torch.randperm(int(evidence_mask.sum(dim=1)[i]))
                # idx = idx.to(device)
                # if evidence_mask.sum(dim=1)[i]<510:
                idx=torch.cat((idx,torch.tensor(range(int(evidence_mask.sum(dim=1)[i]),512)))).to(device)
                new_weights = torch.cat((new_weights.to(device),torch.gather(rationale_weights[i].clone().to(device),0,idx).unsqueeze(0).to(device))).to(device)
                input_s = torch.cat((input_s,torch.gather(input_ids[i].clone(),0,idx).unsqueeze(0))).long().to(device)
            new_weights = new_weights[1:]
            input_s = input_s[1:]
        
            lf_cls_output = self.classify(
            input_ids=input_s,
            mask_weights=new_weights,
            # attention_mask=att_m,
                attention_mask=attention_mask,
            evidence_mask=evidence_mask,
            labels=labels,
            labels_encode = labels_encode,
            pos_encode = pos_encode,
            neg_encode = neg_encode,
            )
            logical_fluency_loss = 1/(1+torch.exp(lf_cls_output.loss+1e-12)) 
            logical_fluency_pred = lf_cls_output.logits.squeeze()
        else:
            logical_fluency_loss =[]
            logical_fluency_pred = []
            for j in range(50):
                new_weights = torch.zeros(1,evidence_mask.shape[1]).to(device)
                input_s = torch.zeros(1,evidence_mask.shape[1]).to(device)
                for i in range(len(evidence_mask)):
                    new_ind = (torch.ones(evidence_mask.shape)*torch.tensor([i for i in range(int(evidence_mask.shape[1]))])).to(device)
                    idx= torch.randperm(int(evidence_mask.sum(dim=1)[i]))
                    idx=torch.cat((idx,torch.tensor(range(int(evidence_mask.sum(dim=1)[i]),512)))).to(device)
                    new_weights = torch.cat((new_weights.to(device),torch.gather(rationale_weights[i].clone().to(device),0,idx).unsqueeze(0).to(device))).to(device)
                    input_s = torch.cat((input_s,torch.gather(input_ids[i].clone(),0,idx).unsqueeze(0))).long().to(device)
                new_weights = new_weights[1:]
                input_s = input_s[1:]
                lf_cls_output = self.classify(
                    input_ids=input_s,
                    mask_weights=new_weights,
                    attention_mask=attention_mask*(new_weights+attention_mask-evidence_mask),
                    evidence_mask = evidence_mask,
                    labels=labels,
                    labels_encode = labels_encode,
                    pos_encode = pos_encode,
                    neg_encode = neg_encode,
                    )
                logical_fluency_loss.append(lf_cls_output.loss.mean().tolist())
                logical_fluency_pred.append(lf_cls_output.logits.tolist())
            logical_fluency_loss= 1/(1+torch.exp(torch.tensor(np.mean(logical_fluency_loss)).to(device)+1e-12)) 
            logical_fluency_pred = torch.tensor(logical_fluency_pred)
        
        loss = ( self.config.suf_weight*suffiency_loss +
                 self.config.cph_weight*cph_loss +
                 self.config.cpt_weight*cpt_loss+
               self.config.mlp_weight*mlp_loss+
               self.config.lf_weight*logical_fluency_loss)


        loss =loss.mean()
        
        rind,rtds = token_dis(r_wd)
        td = (rtds.sum(dim=1)/(evidence_mask.sum(dim=1)+1e-12)).mean()
        
        
        
        lf_loss = logical_fluency_loss.mean()
        lf_loss_gt = torch.tensor([0])
            
        # torch.cuda.empty_cache()
        # print('F：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
        return TokenTaggingRationaleOutput(
            loss=loss,
            # full_logits=full_cls_output.logits,
            masked_logits=masked_cls_output.logits.squeeze(),
            rationale=mask_output.r_wd,
            labels=labels.squeeze(),
            mask_labels=mask_labels.squeeze(),
            # is_unsupervised=is_unsupervised,
            input_ids=input_ids.squeeze(),
            s_loss=suffiency_loss.mean(),
            cph_loss=cph_loss.mean(),
            cpt_loss=cpt_loss.mean(),

            lf_loss=lf_loss,
            lf_loss_gt = lf_loss_gt,
            evidence_len = evidence_mask.sum(dim=1),
            token_dis = td,
            logical_fluency_logits = logical_fluency_pred)
        