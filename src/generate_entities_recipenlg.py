import pandas as pd
import json
import os
from argparse import ArgumentParser
from tqdm import tqdm
from BERT_model import TastyModel
from utils import bio_to_span


def main(args):
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

    print("Prediciton")
    ingredients_entities = model.predict(ingredients)

    print("Converting to spans")
    spans = [bio_to_span(ingredients[idx], ingredients_entities[idx]) for idx in
             range(len(ingredients_entities))]

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
                        help='Path to RecipeNLG')
    script_path = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument('--model-config-path', type=str,
                        default=os.path.join(script_path, "../res/config.json"),
                        help='Path to model config')
    parser.add_argument('--output-path', type=str,
                        default="recipeNLG_with_entities.csv",
                        help="Path under which the file is saved.")
    args = parser.parse_args()
    main(args)

