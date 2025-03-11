from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from transformers.utils import logging
logging.set_verbosity_warning()



def encode_for_training(claims,labels,batch_size,model,positions=None):
    max_source_length = 512
    max_target_length = 7
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    encoding = tokenizer(
        claims,
        padding='max_length',
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
    if positions is not None:
        # print(positions.shape)
        # print(attention_mask.shape)
        attention_mask = attention_mask*positions
    target_encoding = tokenizer(
        labels, padding="longest", max_length=max_target_length, truncation=True
    )
    label_id = target_encoding.input_ids
    label_id = torch.tensor(label_id)
    label_id[label_id == tokenizer.pad_token_id] = -100

    train_dataset = TensorDataset(input_ids, attention_mask, label_id)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader