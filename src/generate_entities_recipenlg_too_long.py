import pandas as pd
import json
import os
from transformers import BertTokenizer
from argparse import ArgumentParser
from tqdm import tqdm

from BERT_model import TastyModel
from utils import bio_to_span, NEWLINE_CHAR


def evaluate_num_of_tokens(some_str):
    input_ids = tokenizer(some_str)["input_ids"]
    return len(input_ids)


def main(args, tokenizer):
    with open(args.model_config_path, "r") as config_file:
        config = json.load(config_file)

    config["model_name_or_path"] = os.path.dirname(args.model_config_path)
    config["bert_type"] = "bert-large-cased"
    model = TastyModel(config=config)

    df = pd.read_csv(args.input_data_path)

    print("Preprocess data")
    df.ingredients = df.ingredients.fillna('')
    df.ingredients = df.ingredients.str.replace(r"\", \"", "\n").str.replace(
        r"(\[|\]|\")", " ").str.strip()
    df.ingredients = df.ingredients.str.replace('\\\\\"', "'")
    df.ingredients = df.ingredients.str.replace(r"\\ ", "'")
    df.ingredients = df.ingredients.str.replace(r"\\t", " ")
    df.ingredients = df.ingredients.str.replace(r"\\n", " ")
    df.ingredients = df.ingredients.str.replace(r"\\u00b0", "°")
    df.ingredients = df.ingredients.str.replace(r" +", ' ')

    ingredients = df.ingredients.to_list()
    ingredients_tokenized = []

    for i, ing in tqdm(enumerate(ingredients), total=len(ingredients)):
        some_list = ing.replace("\n", f" {NEWLINE_CHAR} ").split()
        new_list = []
        el = ""
        for j in range(len(some_list)):
           length = evaluate_num_of_tokens(el + " " + some_list[j])
           if length >= 126:
               new_list.append(
                   el.strip().replace(f" {NEWLINE_CHAR} ", "\n")
                   .replace(f" {NEWLINE_CHAR}", "\n")
                   .replace(f"{NEWLINE_CHAR} ", "\n"))
               el = some_list[j]
           else:
               el = el + " " + some_list[j]
        if el != "":
            new_list.append(
                el.strip().replace(f" {NEWLINE_CHAR} ", "\n")
                .replace(f" {NEWLINE_CHAR}", "\n")
                .replace(f"{NEWLINE_CHAR} ", "\n"))
        
        ingredients_tokenized.append(new_list)
    
    ingredients_entities = []

    print("Prediciton")
    for ing in tqdm(ingredients_tokenized, total=len(ingredients_tokenized)):
        if ing == []:
            ing = [""]
        ing_ents = model.predict(ing)
        ing_ents = [food for i, sublist in enumerate(ing_ents) for food in sublist]
        ingredients_entities.append(ing_ents)

    spans = []
    problematic = []

    print("Converting to spans")
    for idx in tqdm(range(len(ingredients_entities)),
                    total=len(ingredients_entities)):
        try:
            spans.append(bio_to_span(ingredients[idx], ingredients_entities[idx]))
        except:
            spans.append([])
            problematic.append(str(idx)+"\n")

    if len(problematic) == 0:
        print("Great, there are no problematic records!!!")
    else:
        file_name = "problematic.txt"
        print("There are some problematic records, their indices are written "
              f"in {file_name}!!!")
        with open(file_name, "w") as file:
            file.writelines(problematic)

    predictions = dict(zip(df.index.to_list(), spans))

    df["ingredients_entities"] = None

    print("Adding entities to dataframe")
    for idx in tqdm(df.index, total=df.shape[0]):
        entities = predictions[idx]
        ents = "["
        for start, end, entity in entities:
            ents += "{"
            ents += f'"start": {start}, "end": {end}, "type": "{entity}", ' \
                    f'"entity": "{df.ingredients[idx][start:end]}"'
            ents += "},"
        if ents == "[":
            ents += "]"
        else:
            ents = ents[:-1] + "]"

        df.at[idx, "ingredients_entities"] = ents

    print(f"Saving the file under the path: {args.output_path}")
    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input-data-path', type=str,
                        help='Path to RecipeNLG. It is better when it has '
                             'only too long inputs')
    script_path = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument('--model-config-path', type=str,
                        default=os.path.join(script_path, "../res/config.json"),
                        help='Path to model config')
    parser.add_argument('--output-path', type=str,
                        default="recipeNLG_with_entities.csv",
                        help="Path under which the file is saved.")
    args = parser.parse_args()

    tokenizer_path = os.path.dirname(args.model_config_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    main(args, tokenizer)
