import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def token_dis(o_h):
    l_ind = torch.tensor([i for i in range(len(o_h.clone()[0]))]).to(device)
    one_hot_ind = o_h.clone()*l_ind
    ind = torch.nonzero(o_h.clone(),as_tuple=True)[1]
    cot = torch.unique(torch.nonzero(o_h.clone(),as_tuple=True)[0],return_counts=True)[1]
    split = torch.cumsum(cot,0)[:-1].tolist()
    r_inds= torch.tensor_split(ind, split, axis=0)
    r_ind=pad_sequence(r_inds,batch_first=True,padding_value=-1000)  
    pad_ind =  torch.ones(o_h.shape[0],r_ind.shape[-1]).long().to(device)*(-1000)
    pad_ind[o_h.sum(dim=1)>0]=r_ind
    a = abs(F.pad(pad_ind[:,1:],[0,1,0,0],value=-1000)-pad_ind)
    b = abs(pad_ind-F.pad(pad_ind[:,:-1],[1,0,0,0],value=-1000))
    c = torch.min(a,b)
    tds = torch.zeros(o_h.shape[0],c.shape[-1]).long().to(device)
    tds[o_h.sum(dim=1)>1]=c[o_h.sum(dim=1)>1]
    # tds[o_h.sum(dim=1)<=1] = torch.zeros((c[o_h.sum(dim=1)<=1]).shape).long().to(device)
    return pad_ind,tds

def delete(s_v,oh,td_ch,oh_ch):
    pad_ind,tds=token_dis(oh)
    # token_dis = tds.sum(dim=1)
    tdcopy = td_ch.clone()
    oh_d = oh.clone()
    # print(s_v.shape)
    # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
  
    d1_con = oh.sum(dim=1)>2
    if d1_con.sum(dim=0)>0:
    # con11:rationale>1,max_tds=1
        d11_con = ((((tds==torch.max(tds,1)[0].unsqueeze(1)).sum(dim=1))==1)*(d1_con))
        if d11_con.sum(dim=0)>0:
            d11_con_l = torch.tensor([i for i in range(len(d11_con))]).to(device)[d11_con].cpu()
            max_td_v = torch.max(tds,1)[0][d11_con]
            d11_con_c = pad_ind[d11_con][tds[d11_con]==max_td_v.unsqueeze(1)].cpu()
            # d11_con_c = pad_ind[d11_con].gather(1,d11_con_c.unsqueeze(1).to(device)).cpu()
            d11_con_ind = (torch.LongTensor(d11_con_l),torch.LongTensor(d11_con_c))   
            d11_con_val = torch.tensor([0. for _ in range(len(d11_con_l))]).to(device)
            oh_d.index_put_(d11_con_ind,d11_con_val)
            oh_ch_val = torch.tensor([2. for _ in range(len(d11_con_l))]).to(device)
            oh_ch.index_put_(d11_con_ind,oh_ch_val)
            tdd1,tdd_d1 = token_dis(oh_d)
            tdval = (1-(tdd_d1.sum(dim=1)/tds.sum(dim=1)))[d11_con]
            tdcopy.index_put_(d11_con_ind,tdval)

        # con12:rationale>1,max_tds>1
        d12_con = ((((tds==torch.max(tds,1)[0].unsqueeze(1)).sum(dim=1))>1)*(d1_con))
        if d12_con.sum(dim=0)>0:
            d12_con_l = torch.tensor([i for i in range(len(d12_con))]).to(device)[d12_con].cpu()
            inn = pad_ind[d12_con][tds[d12_con]==torch.max(tds,1)[0][d12_con].unsqueeze(1)]
            co = (tds==torch.max(tds,1)[0].unsqueeze(1)).sum(dim=1)[d12_con]
            sp = torch.cumsum(co,0)[:-1].tolist()
            maxtd_inds=torch.tensor_split(inn, sp, axis=0)
            maxtdind_pad=pad_sequence(maxtd_inds,batch_first=True,padding_value=0).to(device)
            # print(d12_con)
            # print(inn)
            # print(sp)
            # print(maxtd_inds)
            # print(maxtdind_pad)
            maxind_val = s_v.clone()[d12_con].gather(1,maxtdind_pad)
            d12_con_c = maxtdind_pad.clone().gather(1,torch.min(-maxind_val,1)[1].unsqueeze(1)).squeeze(0).cpu()
            if len(d12_con_c.shape)>1:
                d12_con_c = d12_con_c.squeeze(1)
            d12_con_ind = (torch.LongTensor(d12_con_l),torch.LongTensor(d12_con_c))
            d12_con_val = torch.tensor([0. for _ in range(len(d12_con_l))]).to(device)
            oh_d.index_put_(d12_con_ind,d12_con_val)
            oh_ch_val = torch.tensor([2. for _ in range(len(d12_con_l))]).to(device)
            oh_ch.index_put_(d12_con_ind,oh_ch_val)
            tdd2,tdd_d2 = token_dis(oh_d)
            tdval2 = (1-(tdd_d2.sum(dim=1)/tds.sum(dim=1)))[d12_con]
            tdcopy.index_put_(d12_con_ind,tdval2)
    return oh_d,oh_ch,tdcopy

def insert(s_v,oh,td_ch,e_m,oh_ch):
    pad_ind,tds=token_dis(oh)
    tdcopy = td_ch.clone()

    # length limitation
    i_l_con = (oh.sum(dim=1)<=e_m.sum(dim=1))
    # sum token-dis condition
    oh_i = oh.clone()
    pad_ind_n,td_n =token_dis(oh_i)
    i_td_con = (td_n.sum(dim=1)>=tds.sum(dim=1))
    # index limitation
    # il = torch.max(pad_ind,1)[0]
    # i_il_cona = il<(e_m.sum(dim=1))-1

    ii=1
    sortgs=s_v.clone().sort(descending=True)[1]
    while (i_td_con*i_l_con).sum(dim=0)>0:
        i_con_c11 = (oh_i.sum(dim=1).long()+ii)
        i_il_con = i_con_c11<(e_m.sum(dim=1))-1
        i_con = i_l_con*i_td_con
        id_con = i_il_con*i_con
        i_con_l=torch.tensor([i for i in range(len(id_con))]).to(device)
        # i_con_c11 = i_con_c11[id_con]
        i_con_c1 = i_con_l.clone()
        i_con_c1[id_con]= torch.gather(sortgs[id_con],1,i_con_c11[id_con].unsqueeze(1)).squeeze(1)
        i_con_c1[i_con_c11>=(e_m.sum(dim=1))-1] = torch.gather(sortgs[i_con_c11>=(e_m.sum(dim=1))-1],1,(oh_i.sum(dim=1).long())[i_con_c11>=(e_m.sum(dim=1))-1].unsqueeze(1)).squeeze(1)
        i_con_l = i_con_l[id_con].cpu()
        i_con_c = i_con_c1[id_con].cpu()
        i_con_ind = (torch.LongTensor(i_con_l),torch.LongTensor(i_con_c))
        i_con_val = torch.tensor([1. for _ in range(id_con.sum(dim=0))]).to(device)
        # print(i_con_ind)
        # print(i_con_val)
        oh_nn = oh_i.clone().index_put_(i_con_ind,i_con_val)
        pad_ind_nn,td_nn =token_dis(oh_nn)
        i_td_con = (td_nn.sum(dim=1)>=tds.sum(dim=1))
        oh_i[i_con*(td_nn.sum(dim=1)<tds.sum(dim=1))] = oh_nn[i_con*(td_nn.sum(dim=1)<tds.sum(dim=1))]
        if (i_con*(td_nn.sum(dim=1)<tds.sum(dim=1))).sum(dim=0)>0:
            i_l = torch.tensor([i for i in range(len(i_con*(td_nn.sum(dim=1)<tds.sum(dim=1))))]).to(device)[(i_con*(td_nn.sum(dim=1)<tds.sum(dim=1)))].cpu()
            i_c = i_con_c1[(i_con*(td_nn.sum(dim=1)<tds.sum(dim=1)))].cpu()
            i_ind = (torch.LongTensor(i_l),torch.LongTensor(i_c))
            oh_ch_val_i = torch.tensor([1. for _ in range(len(i_l))]).to(device)
            oh_ch.index_put_(i_ind,oh_ch_val_i)
            # print(oh_i.shape)
            # print(oh_i[:3])
            # # print(oh_i[3:6])
            # # print(oh_i[6:9])
            tddi,tdd_i = token_dis(oh_i)
            tdval2 = (1-(tdd_i.sum(dim=1)/tds.sum(dim=1)))[i_l]
            tdcopy.index_put_(i_ind,tdval2)
            
            
            # i_l_n = torch.tensor([i for i in range(len(i_con*(td_nn.sum(dim=1)>=tds.sum(dim=1))))]).to(device)[(i_con*(td_nn.sum(dim=1)>=tds.sum(dim=1)))].cpu()
            # i_c_n = i_con_c1[(i_con*(td_nn.sum(dim=1)>=tds.sum(dim=1)))].cpu()
            # i_ind_n = (torch.LongTensor(i_l_n),torch.LongTensor(i_c_n))
            # i_con_val_n = torch.tensor([0. for _ in range(len(i_l_n))]).to(device)
            # oh_nn = oh_nn.index_put_(i_ind_n,i_con_val_n)
            # oh_i[i_con*(td_nn.sum(dim=1)>=tds.sum(dim=1))] = oh_nn[i_con*(td_nn.sum(dim=1)>=tds.sum(dim=1))]

        ii+=1
        if ii>oh_i.shape[-1]*0.1:
            break
        # length limitation
        i_l_con = (oh_i.sum(dim=1)<=e_m.sum(dim=1))
        # sum token-dis condition
        # oh_i = oh.clone()
        pad_ind_n,td_n =token_dis(oh_i)
        i_td_con = (td_n.sum(dim=1)>=tds.sum(dim=1))
        # # index limitation
        # il = torch.max(pad_ind,1)[0]
        # i_il_cona = il<(e_m.sum(dim=1))-1


    return oh_i,oh_ch,tdcopy

def replace(s_v,oh,td_ch,e_m):
    pad_ind,tds=token_dis(oh)
    # oh_ch = oh.clone().fill(0)
    tdcopy = td_ch.clone()

    # Condition1: insert first and delete second
    oh_i1,oh_ch_i1,td_ch_i1 = insert(s_v.clone(),oh.clone(),td_ch,e_m,torch.zeros(oh.shape).to(device))
    oh_d2,oh_ch_d2,td_ch_d2 = delete(s_v.clone(),oh_i1,td_ch_i1,oh_ch_i1)

    # Condition2: delete first and insert second
    oh_d1,oh_ch_d1,td_ch_d1 = delete(s_v.clone(),oh.clone(),td_ch,torch.zeros(oh.shape).to(device))
    oh_i2,oh_ch_i2,td_ch_i2 = insert(s_v.clone(),oh_d1,td_ch_d1,e_m,oh_ch_d1)

    # Select
    s_v_1 = oh_d2*s_v.clone()
    tdd1,t_d_1 = token_dis(oh_d2)
    vd1 = s_v_1.sum(dim=1)/(t_d_1.sum(dim=1)+1e-12)

    s_v_2 = oh_i2*s_v.clone()
    tdd2,t_d_2 = token_dis(oh_i2)
    vd2 = s_v_2.sum(dim=1)/(t_d_2.sum(dim=1)+1e-12)


    oh_r = oh.clone()
    oh_r[vd1>=vd2] = oh_d2[vd1>=vd2]
    oh_r[vd1<vd2] = oh_i2[vd1<vd2]

    oh_ch = torch.zeros(oh.shape).to(device)
    oh_ch[vd1>=vd2] = oh_ch_d2[vd1>=vd2]
    oh_ch[vd1<vd2] = oh_ch_i2[vd1<vd2]

    tdcopy[vd1>=vd2] = td_ch_d2[vd1>=vd2]
    tdcopy[vd1<vd2] = td_ch_i2[vd1<vd2]

    return oh_r,oh_ch,tdcopy

def continuity(s_v,oh,count,e_m):
    con_ra = oh.clone().unsqueeze(1).to(device)
    oh_change =torch.zeros(oh.shape).unsqueeze(1).to(device)
    td_ch = torch.zeros(oh.shape).unsqueeze(1).to(device)

    c = 0
    oh_last = con_ra.clone()
    s_v_last = s_v.clone()

    while c<count:
        oh_l = (oh_last.reshape(-1,oh.shape[-1])).clone().to(device)
        oh_new = torch.zeros(oh_last.shape).to(device)
        oh_ch_new = torch.zeros(oh_last.shape).to(device)
        td_ch_new = torch.zeros(oh_last.shape).to(device)
        td,td_i = token_dis(oh_l.reshape(-1,oh.shape[-1]))


        s_vi = s_v.repeat(int(oh_l.shape[0]/oh.shape[0]),1).clone().to(device)
        tdo = torch.zeros(oh_l.shape).to(device)
        ohch = torch.zeros(oh_l.shape).to(device)
        emo = e_m.repeat(int(oh_l.shape[0]/oh.shape[0]),1).clone().to(device)

        for i in range(3):
            if i == 0:
                oh_d,ohd_ch,oh_ch = delete(s_vi.clone(),oh_l.clone(),tdo.clone(),ohch)
                oh_d = oh_d.unsqueeze(1).reshape(con_ra.shape[0],-1,con_ra.shape[-1])
                ohd_ch = ohd_ch.unsqueeze(1).reshape(con_ra.shape[0],-1,con_ra.shape[-1])
                oh_ch = oh_ch.unsqueeze(1).reshape(con_ra.shape[0],-1,con_ra.shape[-1])
                oh_new =torch.cat((oh_new,oh_d),dim=1)
                oh_ch_new = torch.cat((oh_ch_new,ohd_ch),dim=1)
                td_ch_new = torch.cat((td_ch_new,oh_ch),dim=1)
            elif i==1:
                oh_i,ohi_ch,oh_ch = insert(s_vi.clone(),oh_l.clone(),tdo.clone(),emo.clone(),ohch)
                oh_i = oh_i.unsqueeze(1).reshape(con_ra.shape[0],-1,con_ra.shape[-1])
                ohi_ch = ohi_ch.unsqueeze(1).reshape(con_ra.shape[0],-1,con_ra.shape[-1])
                oh_ch = oh_ch.unsqueeze(1).reshape(con_ra.shape[0],-1,con_ra.shape[-1])
                oh_new =torch.cat((oh_new,oh_i),dim=1)
                oh_ch_new = torch.cat((oh_ch_new,ohi_ch),dim=1)
                td_ch_new = torch.cat((td_ch_new,oh_ch),dim=1)
            else:
                oh_r,ohr_ch,oh_ch = replace(s_vi.clone(),oh_l.clone(),tdo.clone(),emo.clone())
                oh_r = oh_r.unsqueeze(1).reshape(con_ra.shape[0],-1,con_ra.shape[-1])
                ohr_ch = ohr_ch.unsqueeze(1).reshape(con_ra.shape[0],-1,con_ra.shape[-1])
                oh_ch = oh_ch.unsqueeze(1).reshape(con_ra.shape[0],-1,con_ra.shape[-1])
                oh_new =torch.cat((oh_new,oh_r),dim=1)
                oh_ch_new = torch.cat((oh_ch_new,ohr_ch),dim=1)
                td_ch_new = torch.cat((td_ch_new,oh_ch),dim=1)

    
        c+=1
        oh_last = oh_new[:,1:,:]
        con_ra = torch.cat((con_ra,oh_last),dim=1)
        oh_change =torch.cat((oh_change,oh_ch_new[:,1:,:]),dim=1)
        td_ch = torch.cat((td_ch,td_ch_new[:,1:,:]),dim=1)

    return con_ra,td_ch,oh_change

def rl_cd_ss(con_ra,s_v):
    # max_criteria = torch.zeros(con_ra.shape[0],1)
    g_s_n = s_v.clone().unsqueeze(1).repeat(1,con_ra.shape[1],1)
    sv_sum = (g_s_n*con_ra.clone()).sum(dim=2)

    noh = con_ra.reshape(-1,con_ra.shape[-1])
    npad_ind,ntds=token_dis(noh)
    td = ntds.sum(dim=1)
    ntd = td.reshape(-1,con_ra.shape[1])

    cri = sv_sum/(ntd+1e-12)


    r = torch.max(cri,dim=1)[1]
    l = torch.tensor([i for i in range(len(r))]).to(device)

    return r,l