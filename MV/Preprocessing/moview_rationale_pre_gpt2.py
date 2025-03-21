"""Process datasets for token tagging rationale."""

import jsonlines
import logging
import os
import time
import tqdm
import random

from dataclasses import dataclass
from typing import List, Optional, Union
from filelock import FileLock

import torch
from torch.utils.data.dataset import Dataset
from transformers import DataProcessor

from Preprocessing.config_global import global_config
from Preprocessing.movie_processing import download_and_extract
from Preprocessing.connected_shapley_gpt2 import explain_shapley, construct_positions_connectedshapley,explain_shapley_no
# from Preprocessing.sigmoid import sigmoid_function
from util.print_util import iprint

logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class InputExample:
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None
    mask_labels: Optional[List[int]] = None


@dataclass(frozen=True)
class InputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    evidence_mask: Optional[List[float]] = None
    label: Optional[Union[int, float]] = None
    mask_labels: Optional[List[int]] = None
    # is_unsupervised: Optional[float] = 1.0


class RationaleProcessor(DataProcessor):

    @staticmethod
    def _read_jsonlines(input_file):
        lines = []
        with open(input_file, "r", encoding="utf-8") as f:
            reader = jsonlines.Reader(f)
            for line in reader.iter(type=dict):
                lines.append(line)
        # l_len = len(lines)
        # random.seed(1234)
        # line_r = [random.randint(0,l_len-1) for _ in range(int(l_len*0.1))]
        # lines = [lines[i] for i in line_r]
        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonlines(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonlines(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonlines(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["NEG","POS"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if "id" in line:
                guid = line["id"]

            text_a = line["document"]

            if "label" in line:
                label = line["label"]
            else:
                label = None
            
            if "ground-truth_rationale" in line:
                mask_labels = line["ground-truth_rationale"]
            else:
                mask_labels = []

            examples.append(
                {'guid': guid,
                 'text_a': text_a,
                 # 'text_b': text_b,
                 'label': label,
                 'mask_labels': mask_labels})

        return examples


class RationaleDataset(Dataset):

    def __init__(self, args, tokenizer,classifier,neighbor, task_name=None, mode="train", cache_dir=None):
        self.args = args
        self.processor = RationaleProcessor()
        task_name = task_name if task_name else os.path.basename(args.data_dir.rstrip("/"))

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_rationale_{}_{}_{}".format(
                task_name,
                mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
            ),
        )
        self.label_list = self.processor.get_labels()

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        os.makedirs(os.path.dirname(cached_features_file), exist_ok=True)
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                iprint(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                task = "movies"
                task_data_dir = os.path.join(args.data_dir, task)
                if not os.path.exists(task_data_dir):
                    download_and_extract(task, args.data_dir)
                if mode == "dev":
                    examples = self.processor.get_dev_examples(task_data_dir)
                elif mode == "test":
                    examples = self.processor.get_test_examples(task_data_dir)
                elif mode == "train":
                    examples = self.processor.get_train_examples(task_data_dir)
                else:
                    raise ValueError("mode is not a valid split name")

                self.features = convert_fn(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=self.label_list,
                    classifier=classifier,
                    neighbor = neighbor,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # iprint(
                #     "Saving features into cached file %s [took %.3f s]",
                #     cached_features_file, time.time() - start
                # )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.label_list


def convert_fn(examples, tokenizer, label_list,classifier,neighbor, max_length=None, verbose=True):
    """Returns a list of processed features."""
    if max_length is None:
        max_length = tokenizer.max_len

    iprint("Using label list %s" % (label_list))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for example in tqdm.tqdm(examples, "converting to features"):
        mask_labels = []
        all_evidence_tokens = []
        for (i, token) in enumerate(example['text_a'].split()):
            for subtoken in tokenizer.tokenize(token):
                if example['mask_labels']:
                    if i in example['mask_labels']:
                        mask_labels.append(1)
                    else:
                        mask_labels.append(0)
                all_evidence_tokens.append(subtoken)
        all_evidence_tokens = all_evidence_tokens[:min(511,len(all_evidence_tokens))]
        mask_labels = mask_labels[:min(511,len(all_evidence_tokens))]
        # all_claim_tokens = [sub_t for t in example['text_b'].split() for sub_t in tokenizer.tokenize(t)]

        encoding = tokenizer(
            text=' '.join(all_evidence_tokens),
            # text_pair=all_claim_tokens,
            max_length=max_length,
            return_tensors = 'pt',
            padding="max_length",
            truncation=True,
        return_token_type_ids=True)


        num_special = tokenizer.num_special_tokens_to_add(pair=True)
        evidence_mask =  [1.0] * min(max_length - num_special, len(all_evidence_tokens))+[0.0]*(int(encoding['attention_mask'][0].sum(dim=0))-(len(all_evidence_tokens)))
        evidence_mask = [0.0] * (max_length - len(evidence_mask))+ evidence_mask


        if mask_labels:
            # is_unsupervised = 0
            mask_labels = mask_labels+[0]*(int(encoding['attention_mask'][0].sum(dim=0))-(len(all_evidence_tokens)))
            mask_labels = [0]* (max_length - len(mask_labels)) + mask_labels
        # else:
        #     # is_unsupervised = 1
        #     mask_labels = [-1] * max_length
        
        # print(len(mask_labels))
        assert(len(evidence_mask) == max_length)
        assert(len(mask_labels) == max_length)

        if example['label']:
            label = [int(label_map[example['label']])]
            # print(label)
        else:
            print(example)
            label = None
        
        positions_dict, key_to_idx, positions, coefficients, unique_inverse = construct_positions_connectedshapley(int(torch.Tensor(evidence_mask).sum(dim=0)), int(encoding['attention_mask'][0].sum(dim=0)),int(len(encoding['input_ids'][0])),neighbor)
        
        
        input_id = encoding['input_ids'].unsqueeze(0).to(device)
        mask_weights = torch.tensor(positions).unsqueeze(0).to(device)
        mask_ids = input_id.clone().fill_(tokenizer.mask_token_id).to(device)
        
        mix_inputs = (1-mask_weights) * mask_ids +  mask_weights * input_id
        mix_inputs = mix_inputs.long().squeeze().to(device)
    

        atten_m = (encoding['attention_mask'].repeat(len(positions),1)).to(device)*torch.tensor(positions).to(device)
        t_t_i = (encoding['token_type_ids'].repeat(len(positions),1)).to(device)

        
        score = explain_shapley(classifier,int(torch.Tensor(evidence_mask).sum(dim=0)), neighbor, key_to_idx,mix_inputs, encoding['input_ids'],
                                coefficients, unique_inverse,atten_m,t_t_i,encoding['token_type_ids'],encoding['attention_mask'],label)

        # score = sigmoid_function(score)
        # score = torch.nn.functional.softmax(torch.tensor(score).to(device), dim=-1).tolist()
        score = score+[0]*(int(encoding['attention_mask'][0].sum(dim=0))-(len(all_evidence_tokens)))
        score = [0]* (max_length - len(score)) + score

        score1 = explain_shapley_no(classifier, int(torch.Tensor(evidence_mask).sum(dim=0)), neighbor, key_to_idx, mix_inputs,
                                    encoding['input_ids'],
                                    coefficients, unique_inverse, atten_m, t_t_i, encoding['token_type_ids'],
                                    encoding['attention_mask'], label)

        score1 = score1+[0]*(int(encoding['attention_mask'][0].sum(dim=0))-(len(all_evidence_tokens)))
        score1 = [0]* (max_length - len(score1)) + score1
        
        assert(len(score) == max_length)
        assert(len(score1) == max_length)
        
        
        feature = {'input_ids':encoding['input_ids'].squeeze(),
                   'token_type_ids':encoding['token_type_ids'].squeeze(),
                   'attention_mask':encoding['attention_mask'].squeeze(),
            'evidence_mask':torch.Tensor(evidence_mask),
            'scores':torch.Tensor([score,score1]),
            'labels':torch.Tensor(label),
            'mask_labels':torch.Tensor(mask_labels),}


        features.append(feature)

    if verbose:
        for i, example in enumerate(examples[:5]):
            iprint("*** Example ***")
            iprint("guid: %s" % (example['guid']))
            iprint("features: %s" % features[i])

    return features

