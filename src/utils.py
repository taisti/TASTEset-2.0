import re
import pandas as pd
import os
import json
import spacy
from spacy.training import biluo_tags_to_offsets, offsets_to_biluo_tags
from nervaluate import Evaluator


NLP = spacy.load('en_core_web_sm')
ENTITIES = ["FOOD", "QUANTITY", "UNIT", "PROCESS", "PHYSICAL_QUALITY", "COLOR",
            "TASTE", "PURPOSE", "PART", "TRADE_NAME", "DIET", "EXAMPLE"]
NEWLINE_CHAR = "."


def prepare_data(taste_set, entities_format="spans", discontinuous=False):
    """
    :param tasteset: TASTEset as pd.DataFrame or a path to the TASTEset
    :param entities_format: the format of entities. If equal to 'bio', entities
    will be of the following format: [[B-FOOD, I-FOOD, O, ...], [B-UNIT, ...]].
    If equal to span, entities will be of the following format:
    [[(0, 6, FOOD), (10, 15, PROCESS), ...], [(0, 2, UNIT), ...]]
    :param discontinuous: if True, then include discontinuous entites
    :return: list of recipes ingredients and corresponding list of entities
    """

    assert entities_format in ["bio", "spans"],\
        'You provided incorrect entities format!'
    if isinstance(taste_set, pd.DataFrame):
        df = taste_set
    elif isinstance(taste_set, str) and os.path.exists(taste_set):
        df = pd.read_csv(taste_set)
    else:
        raise ValueError('Incorret TASTEset format!')

    all_ingredients = df["ingredients"].to_list()
    all_entities = []

    if discontinuous:
        raise NotImplementedError("The model does not handle discontinuity!")

    for idx in df.index:
        ingredients_entities = json.loads(df.at[idx, "ingredients_entities"])
        entities = []

        for entity_dict in ingredients_entities:
            # pick only specified entities
            if entity_dict["type"] not in ENTITIES:
                continue
            spans = entity_dict["span"]
            spans = re.findall("(\d+, \d+)", spans)
            spans = [[int(char_id) for char_id in span.split(",")] for span
                     in spans]
            for start, end in spans:
                add = True
                # avoid overlapping entities
                for present_start, present_end, _ in entities:
                    if start >= present_start and end <= present_end:
                        add = False
                if add:
                    entities.append((start, end, entity_dict["type"]))

        if entities_format == "bio":
            tokenized_ingredients, entities = span_to_bio(all_ingredients[idx],
                                                          entities)
            tokenized_ingredients = [NEWLINE_CHAR if token == "\n" else token
                    for token in tokenized_ingredients]
            all_ingredients[idx] = tokenized_ingredients

        all_entities.append(entities)

    return all_ingredients, all_entities


def bio_to_biluo(bio_entities):
    """
    :param bio_entities: list of BIO entities, eg. ["O", "B-FOOD", "I-FOOD",
    "B-PROCESS"]
    :return: list of BILUO entities, eg. ["O", "B-FOOD", "L-FOOD", "U-PROCESS"]
    """
    biluo_entities = []

    for entity_idx in range(len(bio_entities)):
        cur_entity = bio_entities[entity_idx]
        next_entity = bio_entities[entity_idx + 1] if \
            entity_idx < len(bio_entities) - 1 else ""

        if cur_entity.startswith("B-"):
            if next_entity.startswith("I-"):
                biluo_entities.append(cur_entity)
            else:
                biluo_entities.append(re.sub("B-", "U-", cur_entity))
        elif cur_entity.startswith("I-"):
            if next_entity.startswith("I-"):
                biluo_entities.append(cur_entity)
            else:
                biluo_entities.append(re.sub("I-", "L-", cur_entity))
        else:  # O
            biluo_entities.append(cur_entity)

    return biluo_entities


def biluo_to_span(ingredients, biluo_entities):
    """
    :param biluo_entities: list of BILUO entities, eg. ["O", "B-FOOD", "L-FOOD",
    "U-PROCESS"]
    :return: list of span entities, eg. [(span_start, span_end, "FOOD"),
    (span_start, span_end, "PROCESS")]
    """
    doc = NLP(ingredients)
    spans = biluo_tags_to_offsets(doc, biluo_entities)
    return spans


def bio_to_span(ingredients, bio_entities):
    """
    :param bio_entities: list of BIO entities, eg. ["O", "B-FOOD", "I-FOOD",
    "B-PROCESS"]
    :return: list of span entities, eg. [(span_start, span_end, "FOOD"),
    (span_start, span_end, "PROCESS")]
    """
    biluo_entities = bio_to_biluo(bio_entities)
    spans = biluo_to_span(ingredients, biluo_entities)
    return spans


def span_to_biluo(ingredients, span_entities):
    """
    :param span_entities: list of span entities, eg. [(span_start, span_end,
    "FOOD"), (span_start, span_end, "PROCESS")]
    :return: list of BILUO entities, eg. ["O", "B-FOOD", "L-FOOD",
    "U-PROCESS"] along with tokenized recipe ingredients
    """
    doc = NLP(ingredients)
    tokenized_ingredients = [token.text for token in doc]
    spans = offsets_to_biluo_tags(doc, span_entities)
    return tokenized_ingredients, spans


def biluo_to_bio(biluo_entities):
    """
    :param biluo_entities: list of BILUO entities, eg. ["O", "B-FOOD", "L-FOOD",
    "U-PROCESS"]
    :return: list of BIO entities, eg. ["O", "B-FOOD", "I-FOOD", "B-PROCESS"]
    """
    bio_entities = [entity.replace("L-", "I-").replace("U-", "B-")
                    for entity in biluo_entities]
    return bio_entities


def span_to_bio(ingredients, span_entities):
    """
    :param span_entities: list of span entities, eg. [(span_start, span_end,
    "FOOD"), (span_start, span_end, "PROCESS")]
    :return: list of BIO entities, eg. ["O", "B-FOOD", "I-FOOD", "B-PROCESS"]
    """
    tokenized_ingredients, biluo_entities = span_to_biluo(ingredients,
                                                          span_entities)
    bio_entities = biluo_to_bio(biluo_entities)
    return tokenized_ingredients, bio_entities


def spans_to_prodigy_spans(list_of_spans):
    """
    Convert to spans format required by nerevaluate.
    """
    prodigy_list_of_spans = []
    for spans in list_of_spans:
        prodigy_spans = []
        for start, end, entity in spans:
            prodigy_spans.append({"label": entity, "start": start, "end": end})
        prodigy_list_of_spans.append(prodigy_spans)

    return prodigy_list_of_spans


def evaluate_predictions(true_entities, pred_entities, entities_format):
    """
    :param true_entities: list of true entities
    :param pred_entities: list of predicted entities
    :param format: format of provided entities. If equal to 'bio', entities
    are expected of the following format: [[B-FOOD, I-FOOD, O, ...],
    [B-UNIT, ...]]. If equal to span, entities are expected of the following
    format: [[(0, 6, FOOD), (10, 15, PROCESS), ...], [(0, 2, UNIT), ...]]
    :return: metrics for the predicted entities
    """

    assert entities_format in ["bio", "spans"],\
        'You provided incorrect entities format!'

    if entities_format == "spans":
        true_entities = spans_to_prodigy_spans(true_entities)
        pred_entities = spans_to_prodigy_spans(pred_entities)

        evaluator = Evaluator(true_entities, pred_entities, tags=ENTITIES)
    else:
        evaluator = Evaluator(true_entities, pred_entities, tags=ENTITIES,
                              loader="list")

    results, results_per_tag = evaluator.evaluate()

    results = results["strict"]

    for entity in results_per_tag.keys():
        results_per_tag[entity] = results_per_tag[entity]["strict"]

    results_per_tag["all"] = results
    return results_per_tag
