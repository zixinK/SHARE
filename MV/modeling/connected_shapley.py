import numpy as np
from tensorflow.keras.utils import to_categorical
import torch

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def construct_positions_connectedshapley(d, k, max_order=None):
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
    values = [np.array(value).reshape(-1, d) for value in values]
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


def explain_shapley(model,d, k, key_to_idx,po_inputs, inputs, coefficients, unique_inverse):
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


    f_vals = []
    # Evaluate predict at inputs.
    # with torch.no_grad():
    for i in range(len(po_inputs)):
        ddd = po_inputs[i].unsqueeze(0).to(device)
        # print(ddd)
        f_result = model(ddd)[1]
        f_result = torch.nn.functional.softmax(f_result, dim=-1).cpu().tolist()
        f_vals.append(f_result)
    del ddd

    p_result = model(inputs.unsqueeze(0))[1]
    probs = torch.nn.functional.softmax(p_result, dim=-1).tolist()  # [1, c]

    log_probs = np.log(f_vals + np.finfo(float).resolution)

    discrete_probs = np.eye(len(probs[0]))[np.argmax(probs, axis=-1)]
    vals = np.sum(discrete_probs * log_probs, axis=1)

    # key_to_idx[key]: list of indices in original position.
    # unique_inverse[idx]: maps idx in original position to idx in the current position.
    key_to_val = {key: np.array([vals[unique_inverse[idx]] for idx in key_to_idx[key]]) for key in key_to_idx}

    # Compute importance scores.
    phis = np.zeros(d)
    for i in range(d):
        phis[i] = np.sum(
            [coefficients[j] * np.sum(key_to_val[(i, j, 1)] - key_to_val[(i, j, 0)]) for j in coefficients])

    return phis
