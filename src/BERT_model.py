import os
import argparse
from transformers import (BertForTokenClassification, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForTokenClassification,
                          set_seed)
from datasets import Dataset
from BERT_with_CRF import BERTCRF
from utils import evaluate_predictions, prepare_data
from BERT_utils import token_to_entity_predictions, tokenize_and_align_labels,\
    prepare_ingredients, CONFIG


class TastyModel:
    def __init__(self, model_name_or_path, config):

        self.config = config
        self.newline_char = self.config["newline_char"]

        self.label2id = {k: int(v) for k, v in self.config["label2id"].items()}
        self.id2label = {v: k for k, v in self.label2id.items()}

        # for reproducibility
        set_seed(self.config["training_args"]["seed"])
        
        tokenizer, model = self.build_model(model_name_or_path)
        
        self.tokenizer = tokenizer
        self.trainer = self.build_trainer(model)
   
    def build_model(self, model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model_class = BERTCRF if self.config["use_crf"] is True else \
            BertForTokenClassification
       
        model = model_class.from_pretrained(
            model_name_or_path,
            num_labels=len(self.config["label2id"]),
            label2id=self.label2id,
            id2label=self.id2label,
            classifier_dropout=self.config['classifier_dropout']
        )

        # add newline token if not present
        if len(tokenizer(self.newline_char, add_special_tokens=False).input_ids) == 0:
            tokenizer.add_tokens(self.newline_char, special_tokens=True)
            model.resize_token_embeddings(len(self.tokenizer))
        
        return tokenizer, model

    def build_trainer(self, model):

        training_args = TrainingArguments(
            **self.config["training_args"]
        )
        
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            max_length=self.config["num_of_tokens"],
            padding="max_length"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        return trainer


    def train(self, train_ingredients, train_entities):

        _, train_dataset = self.prepare_data(train_ingredients, train_entities)

        self.trainer.train_dataset = train_dataset

        self.trainer.train()

    def evaluate(self, ingredients, entities):

        pred_entities = self.predict(ingredients)

        results = evaluate_predictions(entities, pred_entities, "bio")

        return results

    def predict(self, ingredients):

        data, dataset = self.prepare_data(ingredients, [])
        ingredients = data['ingredients']
        
        preds = self.trainer.predict(dataset)
        
        if self.config["use_crf"] is True:
            token_labels = preds[0][1]
        else:
            token_probs = preds[0]
            token_labels = token_probs.argmax(axis=2)

        pred_entities = []

        num_of_recipes = dataset.num_rows

        for recipe_idx in range(num_of_recipes):
            text_split_words = ingredients[recipe_idx]
            text_split_tokens = self.tokenizer.convert_ids_to_tokens(
                data["input_ids"][recipe_idx])

            id2label = self.trainer.model.config.id2label
            if self.config["use_crf"] is True:  # labels are associated to
                # first subwords, hence, are already the word entities
                word_entities = \
                    [self.trainer.model.config.id2label[word_label] for
                     word_label in token_labels[recipe_idx] if word_label != -100]
            else:
                word_entities = token_to_entity_predictions(
                    text_split_words,
                    text_split_tokens,
                    token_labels[recipe_idx],
                    id2label,
                    self.newline_char
                )
            pred_entities.append(word_entities)

        return pred_entities

    def prepare_data(self, ingredients, entities):
        ingredients = prepare_ingredients(ingredients, self.newline_char)
        
        data = tokenize_and_align_labels(
            ingredients=ingredients,
            entities=entities,
            tokenizer=self.tokenizer,
            label2id=self.trainer.model.config.label2id,
            max_length=self.config["num_of_tokens"],
            only_first_token=self.config["only_first_token"]
        )

        dataset = Dataset.from_dict(data)

        return data, dataset

    def save_model(self, trainer):
        save_dir = self.config["save_dir"] if self.config["save_dir"]\
            else "taisti_ner_model"
        os.makedirs(save_dir, exist_ok=True)

        # Add custom config values to the config.json
        self.trainer.model.config.num_of_tokens = self.config["num_of_tokens"]
        self.trainer.model.config.only_first_token = self.config["only_first_token"]
        self.trainer.model.config.training_args = self.config["training_args"]
        self.trainer.model.config.model_pretrained_path = "."
        self.trainer.model.config.use_crf = self.config['use_crf']

        self.trainer.save_model(save_dir)

        print(f"Model with configs saved in {os.path.abspath(save_dir)}!!!")


def train(args):
    model_name_or_path = args.model_name_or_path
    CONFIG["use_crf"] = args.use_crf
    CONFIG["training_args"]["seed"] = args.seed
    CONFIG["save_dir"] = args.save_dir
    CONFIG["newline_char"] = args.newline_char
    bio_ingredients, bio_entities = prepare_data(args.tasteset_path, "bio")

    model = TastyModel(model_name_or_path, config=CONFIG)
    model.train(bio_ingredients, bio_entities)
    model.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model-name-or-path', type=str,
                        help='path to model checkpoint')
    parser.add_argument('--tasteset-path', type=str,
                        default="../data/TASTEset.csv", help="path to TASTEset")
    parser.add_argument('--newline-char', type=str, default="[NL]",
                        help="Token representing new line (instead of \\n)")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for reproducibility")
    parser.add_argument("--use-crf", action='store_true',
                        help="Use CRF layer on top of BERT + linear layer")
    parser.add_argument("--save-dir", type=str, default="tasty_model",
                        help="Path for saving model checkpoint")
    args = parser.parse_args()

    train(args)
