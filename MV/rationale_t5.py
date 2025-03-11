# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning the library models for token-based rationales."""

import logging
import os
import sys

import functools
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

from save_excel import save_excel
from torch.utils.data import DataLoader

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import HfArgumentParser
from transformers import Trainer
from transformers import TrainingArguments
from transformers import set_seed

from Preprocessing.moview_rationale_pre_t5 import RationaleDataset
from modeling.movie_rationale_t5 import (
        T5ForTokenRationale,
        compute_metrics_fn,
        )
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import (
   T5ForConditionalGeneration)


# logging.set_verbosity_warning()

logger = logging.getLogger(__name__)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset name to path.
NAME_TO_DATA_DIR = {
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    num_labels: Optional[int] = field(
        default=3, metadata={"help": "Number of labels (NEI/SUPPORTS/REFUTES)."}
    )
    use_pretrained_classifier: Optional[bool] = field(
        default=False, metadata={"help": "Use a pretrained classifier and keep params fixed."}
    )
    pretrained_classifier_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained classifier model."}
    )
    fix_pretrained_classifier: Optional[bool] = field(
        default=False, metadata={"help": "Do not update classifier parameters."}
    )
    rano_weight: Optional[float] = field(
        default=0.2, metadata={"help": "Lambda for rationale and noise."}
    )
    temperature: Optional[float] = field(
        default=1, metadata={"help": "Gumbel Softmax temperature."}
    )
    # confusion_weight: Optional[float] = field(
    #     default=1.0, metadata={"help": "Lambda for loss margin using masked evidence.."}
    # )
    # continuity_weight: Optional[float] = field(
    #     default=0, metadata={"help": "Lambda for continuity penalty."}
    # )
    suf_weight: Optional[float] = field(
        default=1.0, metadata={"help": "Lambda for suffiency."}
    )
    cph_weight: Optional[float] = field(
        default=0.3, metadata={"help": "Lambda for comprehensive."}
    )
    cpt_weight: Optional[float] = field(
        default=0.1, metadata={"help": "Lambda for compactness."}
    )
    lf_weight: Optional[float] = field(
        default=0.05, metadata={"help": "Lambda for logical fluency."}
    )
    max_order: Optional[int] = field(
        default=16, metadata={"help": "Max order for shapley value."}
    )
    num_neighbors: Optional[int] = field(
        default=4, metadata={"help": "Number of neighbors for shapley value."}
    )
    top_percentage: Optional[float] = field(
        default=0.2, metadata={"help": "Percentage of selected rationale."}
    )
    pro_th: Optional[float] = field(
        default=0.5, metadata={"help": "The threshold of chosen probability."}
    )
    con_count: Optional[int] = field(
        default=3, metadata={"help": "Continuity candidates number."}
    )
    cph_margin: Optional[float] = field(
        default=1, metadata={"help": "The margin of comprehensive."}
    )
    lf_seed: Optional[int] = field(
        default=1234, metadata={"help": "Random seed of logical fluency."}
    )
    lf_num: Optional[int] = field(
        default=1, metadata={"help": "The number of shuffled order."}
    )
    eval_all_checkpoints: Optional[bool] = field(
        default=False, metadata={"help": "Run evaluation on all checkpoints."}
    )
    do_test: Optional[bool] = field(
        default=False, metadata={"help": "Run evaluation on test set (needs labels)."}
    )
    eval_on_datasets: Optional[List[str]] = field(
        default=None, metadata={"help": "List of additional datasets to test on."}
    )
    top_k: Optional[int] = field(
        default=-1, metadata={"help": "Take top-k masked tokens."}
    )
    gen_on_datasets: Optional[List[str]] = field(
        default=None, metadata={"help": "Generate masked data."}
    )


@dataclass
class DataTrainingArguments:
    """
    Data arguments.
    """
    data_dir: str = field(
        default="data/",
        metadata={"help": "The input data dir with jsonl files."}
    )
    data_cache_dir: str = field(
        default="data/cached",
        metadata={"help": "The cache dir for data."}
    )
    max_seq_length: int = field(
        default=256,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
         level=logging.WARN,
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    os.makedirs(training_args.output_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(training_args.output_dir, 'log.txt'))
    logging.getLogger("transformers").setLevel(logging.WARNING)
    fh.setFormatter(formatter)
    logging.getLogger("transformers").addHandler(fh)
    logging.root.addHandler(fh)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Create tokenizer/model.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,  # Our preprocessing is not supported by the fast version.
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=model_args.num_labels,
        cache_dir=model_args.cache_dir,
    )

    # Special args.
    config.nei_index = 2
    config.rano_weight = model_args.rano_weight
    config.temperature = model_args.temperature
    config.suf_weight = model_args.suf_weight
    config.cph_weight = model_args.cph_weight
    config.cpt_weight  = model_args.cpt_weight
    config.mlp_weight = model_args.mlp_weight
    config.lf_weight = model_args.lf_weight
    config.mask_token_id = 32099
    config.use_pretrained_classifier = model_args.use_pretrained_classifier
    config.pretrained_classifier_path = model_args.pretrained_classifier_path
    config.fix_pretrained_classifier = model_args.fix_pretrained_classifier
    # config.unsupervised_weight = model_args.unsupervised_weight
    top_percentage = model_args.top_percentage
    config.pro_th = model_args.pro_th
    pro_th = model_args.pro_th

    # shapley
    config.max_order = model_args.max_order
    config.num_neighbors = model_args.num_neighbors
    config.top_percentage = model_args.top_percentage
    config.con_count = model_args.con_count

    config.cph_margin = model_args.cph_margin
    config.lf_seed = model_args.lf_seed
    config.lf_num = model_args.lf_num
    
    config.testing = model_args.do_test

    model = T5ForTokenRationale.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model.model_parallel =True
    classifier = T5ForConditionalGeneration.from_pretrained(config.pretrained_classifier_path)
    
    # Get our special datasets.
    train_dataset = DataLoader((
        RationaleDataset(data_args, tokenizer=tokenizer,classifier = classifier,neighbor = config.num_neighbors, mode="train", cache_dir=data_args.data_cache_dir)
        if training_args.do_train
        else None
        ),  batch_size=training_args.per_device_train_batch_size)
    eval_dataset = DataLoader((
        RationaleDataset(data_args, tokenizer=tokenizer,classifier = classifier,neighbor = config.num_neighbors, mode="dev", cache_dir=data_args.data_cache_dir)
        if training_args.do_eval
        else None
    ),  batch_size=training_args.per_device_eval_batch_size)
    test_dataset = DataLoader((
        RationaleDataset(data_args, tokenizer=tokenizer,classifier = classifier,neighbor = config.num_neighbors, mode="test", cache_dir=data_args.data_cache_dir)
        if (model_args.do_test or training_args.do_predict)
        else None
    ),  batch_size=training_args.per_device_eval_batch_size)

    # Get extra datasets for testing.
    if model_args.eval_on_datasets:
        extra_test_datasets = {}
        for key in model_args.eval_on_datasets:
            data_args.data_dir = NAME_TO_DATA_DIR[key]
            extra_test_datasets[key] = RationaleDataset(
                data_args, tokenizer=tokenizer, task_name=key, mode="test", cache_dir=data_args.data_cache_dir)
    else:
        extra_test_datasets = {}

    # Get extra datasets for generation.
    if model_args.gen_on_datasets:
        extra_gen_datasets = {}
        for key in model_args.gen_on_datasets:
            data_args.data_dir = NAME_TO_DATA_DIR[key]
            extra_gen_datasets[key] = RationaleDataset(
                data_args, tokenizer=tokenizer, task_name=key, mode="test", cache_dir=data_args.data_cache_dir)
    else:
        extra_gen_datasets = {}

     # Initialize our Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn)

    # --------------------------------------------------------------------------
    # Training.
    # --------------------------------------------------------------------------
    if training_args.do_train:
        optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
        # for name, param in model.named_parameters(): #查看可优化的参数有哪些
              # if param.requires_grad:
              #   logger.info(name)
              #   print(name)
        num_epochs = int(training_args.num_train_epochs)
        num_training_steps = int(num_epochs * len(train_dataset))
        # lr_scheduler = get_scheduler(
        #     "linear",
        #     optimizer=optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=num_training_steps,
        # )
        progress_bar = tqdm(range(num_training_steps))
        model.to(device)
        
        
        train_result = {'Loss':[],
                      "masked_acc": [],
                      "mask_f1": [],
                      "mask_recall":[],
                      "mask_precision": [],
                        "auprc":[],
                        "editing_distance":[],
                        "editing_distance_percentage":[],
                      'LogicalFluency(extracted)': [],
                      'LogicalFluency(ground-truth)': [],
                        'Token_dis':[],
                        # 'lf_acc':[],
                      'Suffiency_loss': [],
                      'Comprehensive_loss': [],
                      'Compactness_loss': [],
                        # "MLP_loss":[],
                      # "Text": [],
                        # 'Score':[]
                          }
        eval_result = {'Loss':[],
                      "masked_acc": [],
                      "mask_f1": [],
                      "mask_recall":[],
                      "mask_precision": [],
                        "auprc":[],
                        "editing_distance":[],
                       "editing_distance_percentage":[],
                      'LogicalFluency_loss(extracted)': [],
                      'LogicalFluency_loss(ground-truth)': [],
                       'Token_dis':[],
                       # 'lf_acc':[],
                      'Suffiency_loss': [],
                      'Comprehensive_loss': [],
                      'Compactness_loss': [],
                       # "MLP_loss":[],
                      # "Text": [],
                       # 'Score':[]
                        }
        # score = []
        min_acc = 0
        min_loss = 100
        for epoch in range(num_epochs):
            loss_s = []
            train_predictions = []
            model.train()
            old_length = 0.4
            for batch in train_dataset:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                # lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                

                outputs = [q.tolist() for p,q in outputs.items()]
                loss_s.append(loss.tolist())
                train_predictions.append(outputs)
                
                # print(outputs)
                model.config.cph_weight = np.mean(outputs[-7])+1e-12/old_length+1e-12
                # old_length = np.mean(outputs[-6])+1e-12
                
                
                # print(outputs)
                # print("======================更新====================")
                # for name, parms in model.named_parameters():	
                #     print('-->name:', name)
                #     print('-->para:', parms)
                #     print('-->grad_requirs:',parms.requires_grad)
                #     print('-->grad_value:',parms.grad)
                #     print("===")

            train_predictions =list(map(list, zip(*train_predictions)))
            # score.append(train_predictions[-1])
            # np.save('{}/train_score.npy'.format(training_args.output_dir), score)
                # print(predictions[0])
            train_r = compute_metrics_fn(train_predictions,tokenizer=tokenizer,return_text=False)
            logger.info("***** Epoch Train Loss *****")
            logger.info("Sum Loss = %s ",sum(loss_s))
            logger.info("Loss = %s ",np.mean(loss_s))

            tran_r = list(train_r.values())
            # print(type(train_predictions[-1]))
            # print(type(train_predictions[-1][0]))
            # tran_r.append(train_predictions[-1])
            tran_r_k = list(train_result.keys())
            # print(len(tran_r_k))
            # print(len(tran_r))
            # print()
            for i in range(len(tran_r_k)):
                val = train_result[tran_r_k[i]]
                val.append(tran_r[i])
                train_result[tran_r_k[i]] = val
            
            
            if training_args.evaluation_strategy == 'epoch':
                model.eval()
                predictions = []
                for batch in eval_dataset:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    outputs = [q.tolist() for p,q in outputs.items()]
                    predictions.append(outputs)
                    # print(outputs)
                predictions =list(map(list, zip(*predictions)))
                # print(predictions[0])
                eval_r = compute_metrics_fn(predictions,tokenizer=tokenizer,return_text=False)

                eval_r = list(eval_r.values())
                # eval_r.append(predictions[-1])
                eval_r_k = list(eval_result.keys())
                for i in range(len(eval_r_k)):
                    val = eval_result[eval_r_k[i]]
                    val.append(eval_r[i])
                    eval_result[eval_r_k[i]] = val
                if eval_r[1]>min_acc:
                    min_acc = eval_r[1]
                    output_model_path = os.path.join(
                        training_args.output_dir, f"result.pt"
                    )
                    torch.save(model.state_dict(), output_model_path)

                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(training_args.output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(training_args.output_dir, CONFIG_NAME)
                    tokenizer.save_vocabulary(training_args.output_dir)

                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
            else:
                if np.mean(loss_s) < min_loss:
                    min_loss = np.mean(loss_s)
                    output_model_path = os.path.join(
                        training_args.output_dir, f"result.pt"
                    )
                    torch.save(model.state_dict(), output_model_path)

                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(training_args.output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(training_args.output_dir, CONFIG_NAME)
                    tokenizer.save_vocabulary(training_args.output_dir)

                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)

        train_result_file_name = f"train_results_t5_suf_{model.config.suf_weight}_cpt_{model.config.cpt_weight}_lf{model.config.lf_weight}.xlsx"  # 动态命名文件
        eval_result_file_name = f"eval_results_t5_suf_{model.config.suf_weight}_cpt_{model.config.cpt_weight}_lf{model.config.lf_weight}.xlsx"  # 动态命名文件
        save_excel(train_result, training_args.output_dir, train_result_file_name)
        save_excel(eval_result, training_args.output_dir, eval_result_file_name)



        # if trainer.is_world_process_zero():
        #     tokenizer.save_pretrained(training_args.output_dir)
    else:
        from transformers import TrainerState
        trainer.state = TrainerState()

    # --------------------------------------------------------------------------
    # Evaluation.
    # --------------------------------------------------------------------------
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        model.eval()
        predictions = []
        for batch in eval_dataset:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            outputs = [q.tolist() for p,q in outputs.items()]
            predictions.append(outputs)
                    # print(outputs)
        predictions =list(map(list, zip(*predictions)))
                # print(predictions[0])
        eval_result = compute_metrics_fn(predictions)
        output_eval_file = os.path.join(
            training_args.output_dir, f"eval_results_k={pro_th}.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        # if model_args.eval_all_checkpoints:
        #     def reinit(cp):
        #         model = AlbertForTokenRationale.from_pretrained(cp, config=config)
        #         trainer = Trainer(
        #             model=model,
        #             args=training_args,
        #             eval_dataset=eval_dataset,
        #             compute_metrics=functools.partial(compute_metrics_fn, tokenizer=tokenizer))
        #         return model, trainer
        #
        #     checkpoints = trainer._sorted_checkpoints()
        #     best_cp = ""
        #     highest_mask_f1 = 0
        #     for cp in checkpoints:
        #         model, trainer = reinit(cp)
        #         eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        #         if trainer.is_world_process_zero():
        #             with open(output_eval_file, "a") as writer:
        #                 logger.info("***** Eval results {} *****".format(cp))
        #                 for key, value in eval_result.items():
        #                     logger.info("  %s = %s", key, value)
        #                     writer.write("%s: %s = %s\n" % (cp, key, value))
        #                     if "mask_f1" in key and value > highest_mask_f1:
        #                         highest_mask_f1 = value
        #                         best_cp = cp
        #
        #     with open(output_eval_file, "a") as writer:
        #         logger.info("***** Best eval mask_f1: {}, {} *****".format(highest_mask_f1, best_cp))
        #         writer.write("best mask_f1: %s (%s)\n" % (highest_mask_f1, best_cp))
        #
        #     model, trainer = reinit(best_cp)

    if model_args.do_test:
        output_model_path = os.path.join(
            model_args.model_name_or_path, f"result.pt"
        )
        model.load_state_dict(torch.load(output_model_path))
        model.eval()
        predictions = []
        for batch in test_dataset:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            outputs = [q.tolist() for p, q in outputs.items()]
            predictions.append(outputs)
            # print(outputs)
        predictions = list(map(list, zip(*predictions)))
        # print(predictions[0])
        test_result = compute_metrics_fn(predictions)
        output_test_file = os.path.join(
            training_args.output_dir, f"test_results_k={pro_th}.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in test_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        # for name, test_dataset in extra_test_datasets.items():
        #     eval_result = trainer.evaluate(eval_dataset=test_dataset)
        #     output_eval_file = os.path.join(
        #         training_args.output_dir, f"test_{name}_results_k={top_percentage}.txt"
        #     )
        #     if trainer.is_world_process_zero():
        #         with open(output_eval_file, "w") as writer:
        #             logger.info("***** Test %s results *****", name)
        #             for key, value in eval_result.items():
        #                 logger.info("  %s = %s", key, value)
        #                 writer.write("%s = %s\n" % (key, value))
        #
        #     if training_args.do_predict:
        #         logging.info("*** Predict ***")
        #         predictions = trainer.predict(test_dataset=test_dataset).predictions
        #         output_pred_file = os.path.join(
        #             training_args.output_dir, f"test_{name}_preds_k={top_percentage}.pt"
        #         )
        #         if trainer.is_world_process_zero():
        #             torch.save(predictions, output_pred_file)

    # if model_args.gen_on_datasets:
    #     for name, gen_dataset in extra_gen_datasets.items():
    #         eval_prediction = trainer.predict(test_dataset=gen_dataset).predictions
    #         mask = eval_prediction[2]
    #         is_masked = np.greater(mask, 0.5).astype(np.int)
    #         input_ids = eval_prediction[6]
    #         examples = []
    #         for i in range(len(input_ids)):
    #             row = tokenizer.convert_ids_to_tokens(input_ids[i])
    #             original = []
    #             masked = []
    #             claim_idx = -1
    #             for j in range(len(row)):
    #                 if row[j] in ["[CLS]", "<pad>"]:
    #                     continue
    #                 if row[j] == "[SEP]":
    #                     claim_idx = j + 1
    #                     break
    #                 original.append(row[j])
    #                 if is_masked[i][j]:
    #                     masked.append("▁<mask>")
    #                 else:
    #                     masked.append(row[j])
    #
    #             claim = []
    #             if claim_idx >= 0:
    #                 for j in range(claim_idx, len(row)):
    #                     if row[j] in ["[CLS]", "<pad>"]:
    #                         continue
    #                     if row[j] == "[SEP]":
    #                         break
    #                     claim.append(row[j])
    #
    #             # clean up.
    #             prev = ""
    #             cleaned = []
    #             for j, token in enumerate(masked):
    #                 if prev == "▁<mask>":
    #                     if not original[j].startswith("▁"):
    #                         continue
    #                 cleaned.append(token)
    #                 prev = token
    #             masked = cleaned
    #
    #             claim = tokenizer.convert_tokens_to_string(claim)
    #             original = tokenizer.convert_tokens_to_string(original)
    #             masked = tokenizer.convert_tokens_to_string(masked)
    #             combined = f"{claim}; EVIDENCE: {original}; PREDS: {masked}"
    #             examples.append(combined)
    #
    #         output_gen_file = os.path.join(
    #             training_args.output_dir, f"test_{name}_gen_k={top_percentage}.source"
    #         )
    #         with open(output_gen_file, "w") as f:
    #             for example in examples:
    #                 f.write(example)
    #                 f.write("\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
