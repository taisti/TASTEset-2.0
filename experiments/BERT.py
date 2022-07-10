import argparse
import json
import numpy as np

from sklearn.model_selection import KFold
from utils import prepare_data, ENTITIES
from BERT_model import TastyModel
from BERT_utils import CONFIG


def cross_validate(args):

    bio_ingredients, bio_entities = prepare_data(args.tasteset_path, "bio")

    model_name_or_path = args.model_name_or_path
    CONFIG["use_crf"] = args.use_crf
    CONFIG["training_args"]["seed"] = args.seed
    CONFIG["newline_char"] = args.newline_char

    kf = KFold(n_splits=args.num_of_folds, shuffle=True, random_state=args.seed)
    cross_val_results = {}

    for fold_id, (train_index, test_index) in enumerate(kf.split(bio_entities)):
        tr_ingredients, vl_ingredients =\
            [bio_ingredients[idx] for idx in train_index],\
            [bio_ingredients[idx] for idx in test_index]
        tr_entities, vl_entities = [bio_entities[idx] for idx in train_index], \
                                   [bio_entities[idx] for idx in test_index]

        model = TastyModel(model_name_or_path, config=CONFIG)
        model.train(tr_ingredients, tr_entities)
        results = model.evaluate(vl_ingredients, vl_entities)
        cross_val_results[fold_id] = results

    with open("bert_cross_val_results.json", "w") as json_file:
        json.dump(cross_val_results, json_file, indent=4)

    # aggregate and print results
    cross_val_results_aggregated = {
        entity: {"precision": [], "recall": [], "f1": []} for entity in
        ENTITIES + ["all"]
    }

    print(f"{'entity':^20s}{'precision':^15s}{'recall':^15s}{'f1-score':^15s}")
    for entity in cross_val_results_aggregated.keys():
        print(f"{entity:^20s}", end="")
        for metric in cross_val_results_aggregated[entity].keys():
            for fold_id in range(args.num_of_folds):
                cross_val_results_aggregated[entity][metric].append(
                    cross_val_results[fold_id][entity][metric]
                )

            mean = np.mean(cross_val_results_aggregated[entity][metric])
            mean = int(mean * 1000) / 1000
            std = np.std(cross_val_results_aggregated[entity][metric])
            std = int(std * 1000) / 1000 + 0.001 * \
                  round(std - int(std * 1000) / 1000)
            print(f"{mean:^2.3f} +- {std:^2.3f} ", end="")
        print()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model-name-or-path', type=str,
                        help='path to model checkpoint')
    parser.add_argument('--tasteset-path', type=str,
                        default="../data/TASTEset.csv", help="path to TASTEset")
    parser.add_argument('--num-of-folds', type=int, default=5,
                        help="Number of folds in cross-validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for reproducibility")
    parser.add_argument("--use-crf", action='store_true',
                        help="Use CRF layer on top of BERT + linear layer")
    parser.add_argument('--newline-char', type=str, default=".",
                        help="Token representing new line (instead of \\n)")

    args = parser.parse_args()

    cross_validate(args)
