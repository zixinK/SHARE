import numpy as np

def newset(r_ind,e_l,s_r,seed,num):
    np.random.seed(seed*num)
    r_index = r_ind
    e_index = [i for i in range(1,e_l+1)]
    # n_e_rindex = [i for i in range(1,e_l) if i not in r_index]
    ne_index = [i for i in range(e_l+1,len(s_r))]
    r_len = len(r_index)
    d_len = len(s_r)
    count = 0
    new_word_pos = []

    # while count<num:
    str =''
    word_pos_arr = [[i, 0] for i in range(d_len)]
    word_pos_arr[0][1] = word_pos_arr[0][0]
    pos_arr = np.random.choice(e_index,r_len, False)
    print(len(pos_arr))
    print(len(r_index))
    print(r_index)
    print(len(e_index))
    for i in range(r_len):
        word_pos_arr[r_index[i]][1] = pos_arr[i]
    nind = [i for i in range(1,e_l+1) if i not in r_index]
    pind = [i for i in range(1,e_l+1) if i not in pos_arr]
    print(len(nind))
    print(len(pind))
    assert(len(nind) == len(pind))
    for j in range(len(nind)):
        word_pos_arr[nind[j]][1] = pind[j]
    for j in range(e_l+1,d_len):
        word_pos_arr[j][1] = word_pos_arr[j][0]
    word_pos_arr = sorted(word_pos_arr,key=lambda item:item[1])
    w_p = [x[0] for x in  word_pos_arr]
        # new_word_pos.append(w_p)
        # count +=1
    return w_p

