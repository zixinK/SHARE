import numpy as np
from tensorflow.keras.utils import to_categorical
import torch
import time
# from memory_profiler import profile
import math
from transformers.utils import logging
logging.set_verbosity_warning()
logger = logging.get_logger("transformers")
logger.warning("WARN")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# @profile
def construct_positions_connectedshapley(d, d_s, k, max_order=None):
    # Construct collection of subsets for each feature.
    subsets = {}

    while k >= d:
        k -= 2

    if max_order == None:
        max_order = k + 1

    # coefficients.
    coefficients = {j: 2.0 / (j * (j + 1) * (j + 2)) for j in range(1, max_order + 1)}
    # max_order = 4 # k+1
    for i in range(d):
        subsets[i] = [np.array(range(i - s, i + t + 1)) % d for s in range(k // 2 + 1) for t in range(k // 2 + 1)]
        subsets[i] = [subset for subset in subsets[i] if len(subset) <= max_order]
    # Construct dictionary of indices for points where
    # the predict is to be evaluated.
    positions_dict = {(i, l, fill): [] for i in range(d) for l in range(1, max_order + 1) for fill in [0, 1]}
    for i in subsets:
        # pad positions outside S by 1.
        one_pads = 1 - np.sum(to_categorical(np.array(range(i - k // 2, i + k // 2 + 1)) % d, num_classes=d), axis=0)

        for arr in subsets[i]:
            # For feature i, the list of subsets of size j, with/without feature i.
            l = len(arr)
            pos_included = np.sum(to_categorical(arr, num_classes=d), axis=0)
            # pad positions outside S by 1.
            pos_included += one_pads
            pos_excluded = pos_included - to_categorical(i, num_classes=d)
            positions_dict[(i, l, 1)].append(pos_included)
            positions_dict[(i, l, 0)].append(pos_excluded)

    # values is a list of lists of zero-one vectors.
    keys, values = positions_dict.keys(), positions_dict.values()

    # concatenate a list of lists to a list.
    # print(values)
    values = [
        np.pad(np.array(value[0].tolist()), (0, d_s - d), 'constant', constant_values=(0, 1)).reshape(
            -1, d_s) for value in values]
    # print(values)

    positions = np.concatenate(values, axis=0)

    key_to_idx = {}
    count = 0
    for i, key in enumerate(keys):
        key_to_idx[key] = list(range(count, count + len(values[i])))
        count += len(values[i])

    # reduce the number of func evaluation by removing duplicate.
    positions, unique_inverse = np.unique(positions, axis=0, return_inverse=True)
    # positions = 1 - positions

    return positions_dict, key_to_idx, positions, coefficients, unique_inverse


def explain_shapley(model, d, k, key_to_idx, coefficients, unique_inverse, 
                    train_dataloader,train_dataloader1, label,tokenizer,pos_encoding,neg_encoding):

    while k >= d:
        k -= 2

    model = model.to(device)
    f_vals=[]
    if label == 'POS':
        label_id = 1
    else:
        label_id=0

    with torch.no_grad():
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            
            label_id_pos_t = pos_encoding.repeat(batch[2].shape[0],1).to(device)
            label_id_neg_t = neg_encoding.repeat(batch[2].shape[0],1).to(device)
            
            lgp = torch.prod(torch.nn.functional.softmax(outputs.logits, dim=-1).gather(2,label_id_pos_t.unsqueeze(2)),1)
            lgn = torch.prod(torch.nn.functional.softmax(outputs.logits, dim=-1).gather(2,label_id_neg_t.unsqueeze(2)),1)
            if len(lgp.shape) !=2:
                lgp = lgp.unsqueeze(0)
            if len(lgn.shape) !=2:
                lgn = lgn.unsqueeze(0)
            logit = torch.cat((lgn,lgp),1).tolist()
            f_vals.extend(logit)

    with torch.no_grad():
        for batch in train_dataloader1:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            probs = [0,0]
            
            label_id_pos_t = pos_encoding.repeat(batch[2].shape[0],1).to(device)
            label_id_neg_t = neg_encoding.repeat(batch[2].shape[0],1).to(device)
            
            probs[0] = torch.prod(torch.nn.functional.softmax(outputs.logits, dim=-1).gather(2,label_id_neg_t.unsqueeze(2))).tolist()
            probs[1] =torch.prod(torch.nn.functional.softmax(outputs.logits, dim=-1).gather(2,label_id_pos_t.unsqueeze(2))).tolist()

    # print(len(f_vals))
    # print(f_vals[0])
    # print('???????????????????????????????????????????????')
    log_probs = np.log(f_vals + np.finfo(float).resolution)

    # discrete_probs = np.eye(len(probs[0]))[np.argmax(probs, axis=-1)]
    discrete_probs = np.eye(len(probs))[int(label_id)]
    # print(discrete_probs * log_probs)
    vals = np.sum(discrete_probs * log_probs,axis = 1)

    # key_to_idx[key]: list of indices in original position.
    # unique_inverse[idx]: maps idx in original position to idx in the current position.
    key_to_val = {key: np.array([vals[unique_inverse[idx]] for idx in key_to_idx[key]]) for key in key_to_idx}

    # Compute importance scores.
    phis = [0 for i in range(d)]
    for i in range(d):
        phis[i] = np.sum(
            [coefficients[j] * np.sum(key_to_val[(i, j, 1)] - key_to_val[(i, j, 0)]) for j in coefficients])
    return phis


def explain_shapley_no(model, d, k, key_to_idx, coefficients, unique_inverse, 
                    train_dataloader,train_dataloader1, label,tokenizer,pos_encoding,neg_encoding):
    """
    Compute the importance score of each feature of x for the predict.

    Inputs:
    predict: a function that takes in inputs of shape (n,d), and
    outputs the distribution of response variable, of shape (n,c),
    where n is the number of samples, d is the input dimension, and
    c is the number of classes.

    x: input vector (d,)

    k: number of neighbors taken into account for each feature.

    Outputs:
    phis: importance scores of shape (d,)
    """
    while k >= d:
        k -= 2

    model = model.to(device)
    # f_vals = []
    # Evaluate predict at inputs.
    # start1=time.time()
    # epo = math.ceil(len(po_inputs) / 512)
    f_vals = []
    if label == 'POS':
        label_id = 1
    else:
        label_id=0
    # print(po_inputs.shape)
    with torch.no_grad():
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            
            label_id_pos_t = pos_encoding.repeat(batch[2].shape[0],1).to(device)
            label_id_neg_t = neg_encoding.repeat(batch[2].shape[0],1).to(device)
            
            lgp = torch.prod(torch.nn.functional.softmax(outputs.logits, dim=-1).gather(2,label_id_pos_t.unsqueeze(2)),1)
            lgn = torch.prod(torch.nn.functional.softmax(outputs.logits, dim=-1).gather(2,label_id_neg_t.unsqueeze(2)),1)
            if len(lgp.shape) !=2:
                lgp = lgp.unsqueeze(0)
            if len(lgn.shape) !=2:
                lgn = lgn.unsqueeze(0)
            logit = torch.cat((lgn,lgp),1).tolist()
            f_vals.extend(logit)

    with torch.no_grad():
        for batch in train_dataloader1:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            probs = [0,0]
            
            label_id_pos_t = pos_encoding.repeat(batch[2].shape[0],1).to(device)
            label_id_neg_t = neg_encoding.repeat(batch[2].shape[0],1).to(device)
            
            probs[0] = torch.prod(torch.nn.functional.softmax(outputs.logits, dim=-1).gather(2,label_id_neg_t.unsqueeze(2))).tolist()
            probs[1] =torch.prod(torch.nn.functional.softmax(outputs.logits, dim=-1).gather(2,label_id_pos_t.unsqueeze(2))).tolist()

    log_probs = np.log(f_vals + np.finfo(float).resolution)

    # discrete_probs = np.eye(len(probs[0]))[np.argmax(probs, axis=-1)]
    discrete_probs = 1 - np.eye(len(probs))[int(label_id)]
    vals = np.sum(discrete_probs * log_probs,axis = 1)

    # key_to_idx[key]: list of indices in original position.
    # unique_inverse[idx]: maps idx in original position to idx in the current position.
    key_to_val = {key: np.array([vals[unique_inverse[idx]] for idx in key_to_idx[key]]) for key in key_to_idx}

    # Compute importance scores.
    phis = [0 for i in range(d)]
    for i in range(d):
        phis[i] = np.sum(
            [coefficients[j] * np.sum(key_to_val[(i, j, 1)] - key_to_val[(i, j, 0)]) for j in coefficients])
    return phis