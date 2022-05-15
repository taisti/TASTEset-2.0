#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning (m)LUKE model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library 🤗
without using a Trainer.
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path

import datasets
import torch
from datasets import ClassLabel, load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from huggingface_hub import Repository
from luke_utils import DataCollatorForLukeTokenClassification, is_punctuation, padding_tensor
from transformers import (
    AdamW,
    LukeConfig,
    LukeForEntitySpanClassification,
    LukeTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
import numpy as np
from utils import prepare_data, evaluate_predictions, ENTITIES
from sklearn.model_selection import KFold
import os
os.environ["NCCL_DEBUG"] = "INFO"

SEED = 42
NUM_OF_FOLDS = 5
logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune (m)LUKE on a token classification task (such as NER) with the accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--max_entity_length",
        type=int,
        default=512,
        help=(
            "The maximum total input entity length after tokenization (Used only for (M)Luke models). Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--max_mention_length",
        type=int,
        default=5,
        help=(
            "The maximum total input mention length after tokenization (Used only for (M)Luke models). Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=30, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=SEED, help="A seed for reproducible training.")
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


class LukeDataset():
    def __init__(self, tokens, entities, padding):
        self.text_column_name = "tokens"
        self.label_column_name = "ner_tags"
        self.padding = padding
        self.max_length = 1024
        self.tokenizer = None
        self.tokens = tokens
        self.entities = [[el.replace("B-", "").replace("I-", "") for el in elem] for elem in entities]
        self.unique_tag_names = ["O"] + ENTITIES
        self.sentence_boundaries = []
        self.tokenized_input = []
        self.entity_spans = []
        self.text = []
        self.labels_entity_spans = []
        self.original_entity_spans = []

        self.ner_tags = self.generate_ner_tags()



    def generate_ner_tags(self):
        d = {}
        for idx, ent in enumerate(self.unique_tag_names):
            d[ent] = idx
        ner_tags = [[d[e] for e in ent]for ent in self.entities]

        return ner_tags

    def compute_sentence_boundaries_for_luke(self):
        for tokens in self.tokens:
            self.sentence_boundaries.append([0, len(tokens)])


    def compute_entity_spans_for_luke(self):
        all_entity_spans = []
        texts = []
        all_labels_entity_spans = []
        all_original_entity_spans = []

        for labels, tokens, sentence_boundaries in zip(
            self.ner_tags, self.tokens, self.sentence_boundaries
        ):
            subword_lengths = [len(self.tokenizer.tokenize(token)) for token in tokens]
            total_subword_length = sum(subword_lengths)
            _, context_end = sentence_boundaries

            if total_subword_length > self.max_length - 2:
                cur_length = sum(subword_lengths[:context_end])
                idx = context_end - 1

                while cur_length > self.max_length - 2:
                    cur_length -= subword_lengths[idx]
                    context_end -= 1
                    idx -= 1

            text = ""
            sentence_words = tokens[:context_end]
            sentence_subword_lengths = subword_lengths[:context_end]
            word_start_char_positions = []
            word_end_char_positions = []
            labels_positions = {}

            for word, label in zip(sentence_words, labels):
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()

                word_start_char_positions.append(len(text))
                text += word
                word_end_char_positions.append(len(text))
                text += " "
                labels_positions[(word_start_char_positions[-1], word_end_char_positions[-1])] = label

            # text = text.rstrip()
            # texts.append(text.replace(" \n ", "\n").replace("\n ", "\n").replace(" \n", "\n"))

            texts.append(text)
            entity_spans = []
            labels_entity_spans = []
            original_entity_spans = []

            for word_start in range(len(sentence_words)):
                for word_end in range(word_start, len(sentence_words)):
                    if (
                        sum(sentence_subword_lengths[word_start:word_end]) <= self.tokenizer.max_mention_length
                        and len(entity_spans) < self.tokenizer.max_entity_length
                    ):
                        entity_spans.append((word_start_char_positions[word_start], word_end_char_positions[word_end]))
                        original_entity_spans.append((word_start, word_end + 1))
                        if (
                            word_start_char_positions[word_start],
                            word_end_char_positions[word_end],
                        ) in labels_positions:
                            labels_entity_spans.append(
                                labels_positions[
                                    (word_start_char_positions[word_start], word_end_char_positions[word_end])
                                ]
                            )
                        else:
                            labels_entity_spans.append(0)

            all_entity_spans.append(entity_spans)
            all_labels_entity_spans.append(labels_entity_spans)
            all_original_entity_spans.append(original_entity_spans)

        self.entity_spans = all_entity_spans
        self.text = texts
        self.labels_entity_spans = all_labels_entity_spans
        self.original_entity_spans = all_original_entity_spans


    def tokenize_and_align_labels(self):
        entity_spans = []

        for v in self.entity_spans:
            entity_spans.append(list(map(tuple, v)))

        for idx, elem in enumerate(entity_spans):
            if len(elem) == 0:
                entity_spans[idx] = list(map(tuple, [[0, 0]]))

        tokenized_inputs = self.tokenizer(
            self.text,
            entity_spans=entity_spans,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
        )
        if self.padding == "max_length":
            tokenized_inputs["labels"] = padding_tensor(
                self.labels_entity_spans, -100, self.tokenizer.padding_side, self.tokenizer.max_entity_length
            )
            tokenized_inputs["original_entity_spans"] = padding_tensor(
                self.original_entity_spans, (-1, -1), self.tokenizer.padding_side, self.tokenizer.max_entity_length
            )
            tokenized_inputs[self.label_column_name] = padding_tensor(
                self.ner_tags, -1, self.tokenizer.padding_side, self.tokenizer.max_entity_length
            )
        else:
            tokenized_inputs["labels"] = [ex[: self.tokenizer.max_entity_length] for ex in
                                          self.labels_entity_spans]
            tokenized_inputs["original_entity_spans"] = [
                ex[: self.tokenizer.max_entity_length] for ex in self.original_entity_spans
            ]
            tokenized_inputs[self.label_column_name] = [
                ex[: self.tokenizer.max_entity_length] for ex in self.ner_tags
            ]

        return tokenized_inputs


    def compute_rest(self, tokenizer):
        self.tokenizer = tokenizer
        self.compute_sentence_boundaries_for_luke()
        self.compute_entity_spans_for_luke()
        self.tokenized_input = self.tokenize_and_align_labels()





recipes, entities = prepare_data("../data/TASTEset.csv", "bio")

kf = KFold(n_splits=NUM_OF_FOLDS, shuffle=True, random_state=SEED)



def main():
    all_kfold_eval_metrics = []

    for fold_id, (train_index, test_index) in enumerate(kf.split(entities)):

        args = parse_args()

        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[handler])
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state)

        # Setup logging, we only want one process per machine to log things on the screen.
        # accelerator.is_local_main_process is only True for one process per machine.
        logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Handle the repository creation
        if accelerator.is_main_process:
            if args.push_to_hub:
                if args.hub_model_id is None:
                    repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
                else:
                    repo_name = args.hub_model_id
                repo = Repository(args.output_dir, clone_from=repo_name)
            elif args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
        # (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
        # 'tokens' is found. You can easily tweak this behavior (see below).
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        padding = "max_length" if args.pad_to_max_length else False

        train_recipes, test_recipes = [recipes[idx] for idx in train_index], \
                                      [recipes[idx] for idx in test_index]
        train_entities, test_entities = [entities[idx] for idx in train_index], \
                                        [entities[idx] for idx in test_index]
        #percentile of not annotated data:
        # [el for elem in all_entities for el in elem].count("O") / len(
        #     [el for elem in all_entities for el in elem]) * 100

        raw_datasets = datasets.DatasetDict({"train": LukeDataset(train_recipes, train_entities, padding),
                                             "test": LukeDataset(test_recipes, test_entities, padding)})
        # if args.dataset_name is not None:
        #     # Downloading and loading a dataset from the hub.
        #     # raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        # else:
        #     data_files = {}
        #     if args.train_file is not None:
        #         data_files["train"] = args.train_file
        #     if args.validation_file is not None:
        #         data_files["validation"] = args.validation_file
        #     extension = args.train_file.split(".")[-1]
        #     raw_datasets = load_dataset(extension, data_files=data_files)
        # Trim a number of training examples
        # if args.debug:
        #     for split in raw_datasets.keys():
        #         raw_datasets[split] = raw_datasets[split].select(range(100))
        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.


        # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
        # unique labels.
        # def get_label_list(labels):
        #     unique_labels = set()
        #     for label in labels:
        #         unique_labels = unique_labels | set(label)
        #     label_list = list(unique_labels)
        #     label_list.sort()
        #     return label_list

        # if isinstance(features[label_column_name].feature, ClassLabel):
        #     label_list = features[label_column_name].feature.names
        #     # No need to convert the labels since they are already ints.
        # else:
        #     label_list = get_label_list(raw_datasets["train"][label_column_name])

        label_list = raw_datasets["train"].unique_tag_names
        num_labels = len(label_list)

        # Map that sends B-Xxx label to its I-Xxx counterpart
        b_to_i_label = []

        for idx, label in enumerate(label_list):
            if label.startswith("B-") and label.replace("B-", "I-") in label_list:
                b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
            else:
                b_to_i_label.append(idx)

        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        if args.config_name:
            config = LukeConfig.from_pretrained(args.config_name, num_labels=num_labels)
        elif args.model_name_or_path:
            config = LukeConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        else:
            logger.warning("You are instantiating a new config instance from scratch.")

        tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
        if not tokenizer_name_or_path:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

        tokenizer = LukeTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=False,
            task="entity_span_classification",
            max_entity_length=args.max_entity_length,
            max_mention_length=args.max_mention_length,
        )

        raw_datasets["train"].compute_rest(tokenizer)
        raw_datasets["test"].compute_rest(tokenizer)

        if args.model_name_or_path:
            model = LukeForEntitySpanClassification.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
        else:
            logger.info("Training new model from scratch")
            model = LukeForEntitySpanClassification.from_config(config)

        model.resize_token_embeddings(len(tokenizer))


        train_dataset = [{"input_ids":list(raw_datasets["train"].tokenized_input.values())[0][el], 'entity_ids':list(raw_datasets["train"].tokenized_input.values())[1][el], 'entity_position_ids':list(raw_datasets["train"].tokenized_input.values())[2][el], 'entity_start_positions':list(raw_datasets["train"].tokenized_input.values())[3][el], 'entity_end_positions':list(raw_datasets["train"].tokenized_input.values())[4][el], 'attention_mask':list(raw_datasets["train"].tokenized_input.values())[5][el], 'entity_attention_mask':list(raw_datasets["train"].tokenized_input.values())[6][el], 'labels':list(raw_datasets["train"].tokenized_input.values())[7][el], 'original_entity_spans':list(raw_datasets["train"].tokenized_input.values())[8][el], 'ner_tags':list(raw_datasets["train"].tokenized_input.values())[9][el]} for el in range(0,len(list(raw_datasets["train"].tokenized_input.values())[0]))]
        eval_dataset = [{"input_ids":list(raw_datasets["test"].tokenized_input.values())[0][el], 'entity_ids':list(raw_datasets["test"].tokenized_input.values())[1][el], 'entity_position_ids':list(raw_datasets["test"].tokenized_input.values())[2][el], 'entity_start_positions':list(raw_datasets["test"].tokenized_input.values())[3][el], 'entity_end_positions':list(raw_datasets["test"].tokenized_input.values())[4][el], 'attention_mask':list(raw_datasets["test"].tokenized_input.values())[5][el], 'entity_attention_mask':list(raw_datasets["test"].tokenized_input.values())[6][el], 'labels':list(raw_datasets["test"].tokenized_input.values())[7][el], 'original_entity_spans':list(raw_datasets["test"].tokenized_input.values())[8][el], 'ner_tags':list(raw_datasets["test"].tokenized_input.values())[9][el]} for el in range(0,len(list(raw_datasets["test"].tokenized_input.values())[0]))]

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        # DataLoaders creation:
        if args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            data_collator = DataCollatorForLukeTokenClassification(
                tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
            )

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Use the device given by the `accelerator` object.
        device = accelerator.device
        model.to(device)

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # Metrics
        metric = load_metric("seqeval")

        def get_luke_labels(outputs, ner_tags, original_entity_spans):
            true_predictions = []
            true_labels = []

            for output, original_spans, tags in zip(outputs.logits, original_entity_spans, ner_tags):
                true_tags = [val for val in tags if val != -1]
                true_original_spans = [val for val in original_spans if val != (-1, -1)]
                max_indices = torch.argmax(output, axis=1)
                max_logits = torch.max(output, axis=1).values
                predictions = []

                for logit, index, span in zip(max_logits, max_indices, true_original_spans):
                    if index != 0:
                        predictions.append((logit, span, label_list[index]))

                predicted_sequence = [label_list[0]] * len(true_tags)

                for _, span, label in sorted(predictions, key=lambda o: o[0], reverse=True):
                    if all([o == label_list[0] for o in predicted_sequence[span[0] : span[1]]]):
                        predicted_sequence[span[0]] = label
                        if span[1] - span[0] > 1:
                            predicted_sequence[span[0] + 1 : span[1]] = [label] * (span[1] - span[0] - 1)

                true_predictions.append(predicted_sequence)
                true_labels.append([label_list[tag_id] for tag_id in true_tags])

            return true_predictions, true_labels

        def compute_metrics():
            results = metric.compute()
            if args.return_entity_level_metrics:
                # Unpack nested dictionaries
                final_results = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        for n, v in value.items():
                            final_results[f"{key}_{n}"] = v
                    else:
                        final_results[key] = value
                return final_results
            else:
                return {
                    "precision": results["overall_precision"],
                    "recall": results["overall_recall"],
                    "f1": results["overall_f1"],
                    "accuracy": results["overall_accuracy"],
                    'f1-food': results["OOD"]["f1"],
                    'f1-quantity': results["UANTITY"]["f1"],
                    'f1-unit': results["NIT"]["f1"],
                    'f1-process': results["ROCESS"]["f1"],
                    'f1-physical_quality': results["HYSICAL_QUALITY"]["f1"],
                    'f1-color': results["OLOR"]["f1"],
                    'f1-taste': results["ASTE"]["f1"],
                    'f1-purpose': results["URPOSE"]["f1"],
                    'f1-part': results["ART"]["f1"],
                    'f1-trade-name': results["RADE_NAME"]["f1"],
                    'f1-diet': results["IET"]["f1"],
                    'f1-example': results["XAMPLE"]["f1"]
                }

        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                _ = batch.pop("original_entity_spans")
                _ = batch.pop("ner_tags")
                outputs = model(input_ids=batch["input_ids"], entity_ids=batch['entity_ids'], entity_position_ids=batch['entity_position_ids'], entity_start_positions=batch['entity_start_positions'], entity_end_positions=batch['entity_end_positions'], attention_mask=batch['attention_mask'], entity_attention_mask=batch['entity_attention_mask'], labels=batch['labels'])
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break

            # model.eval()
            # for step, batch in enumerate(eval_dataloader):
            #     original_entity_spans = batch.pop("original_entity_spans")
            #     ner_tags = batch.pop("ner_tags")
            #     with torch.no_grad():
            #         outputs = model(input_ids=batch["input_ids"], entity_ids=batch['entity_ids'], entity_position_ids=batch['entity_position_ids'], entity_start_positions=batch['entity_start_positions'], entity_end_positions=batch['entity_end_positions'], attention_mask=batch['attention_mask'], entity_attention_mask=batch['entity_attention_mask'], labels=batch['labels'])
            #
            #     preds, refs = get_luke_labels(outputs, ner_tags, original_entity_spans)
            #
            #     metric.add_batch(
            #         predictions=preds,
            #         references=refs,
            #     )  # predictions and preferences are expected to be a nested list of labels, not label_ids
            #
            # eval_metric = compute_metrics()
            # accelerator.print(f"epoch {epoch}:", eval_metric)


        model.eval()
        for step, batch in enumerate(eval_dataloader):
            original_entity_spans = batch.pop("original_entity_spans")
            ner_tags = batch.pop("ner_tags")
            with torch.no_grad():
                outputs = model(input_ids=batch["input_ids"], entity_ids=batch['entity_ids'],
                                entity_position_ids=batch['entity_position_ids'],
                                entity_start_positions=batch['entity_start_positions'],
                                entity_end_positions=batch['entity_end_positions'],
                                attention_mask=batch['attention_mask'],
                                entity_attention_mask=batch['entity_attention_mask'], labels=batch['labels'])

            preds, refs = get_luke_labels(outputs, ner_tags, original_entity_spans)

            metric.add_batch(
                predictions=preds,
                references=refs,
            )  # predictions and preferences are expected to be a nested list of labels, not label_ids

        eval_metric = compute_metrics()
        accelerator.print(f"epoch {epoch}:", eval_metric)
        all_kfold_eval_metrics.append(eval_metric)

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                if args.push_to_hub:
                    repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    for score in all_kfold_eval_metrics[0].keys():
        score_val = [elem[score] for elem in all_kfold_eval_metrics]
        print(f"{score}, avg: {np.mean(score_val)}, std: {np.std(score_val)}")

if __name__ == "__main__":
    main()
