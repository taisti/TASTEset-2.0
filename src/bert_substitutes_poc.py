from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

from BERT_utils import prepare_ingredients_for_prediction


MODEL_PATH = "../res"
RECIPE = "2 eggs\n1 tablespoon milk\n1 teaspoon butter\n1/2 avocado (diced)" \
         "\n1 teaspoon chives (chopped)"
RECIPE_WITH_SUBSTITUTE = RECIPE.replace("butter", "margarine")
QUERY_WORD = "butter"
POTENTIAL_SUBSTITUTE = "margarine" 
WRONG_SUBSTITUTE = "sour cream"
WRONG_SUBSTITUTE2 = "peanut butter"
VERY_WRONG_SUBSTITUTE = "chicken"
VERY_WRONG_SUBSTITUTE2 = "potato"


def locate_word(tokenized_word, tokenized_recipe):

    start_idx = 0
    end_idx = 0

    for idx in range(len(tokenized_recipe)):
        if tokenized_recipe[idx] == tokenized_word[0]:
            i = 1
            while i < len(tokenized_word):
                if tokenized_word[i] != tokenized_recipe[idx + i]:
                    break

                i += 1

            if i == len(tokenized_word):
                start_idx = idx
                end_idx = idx + i

    return start_idx, end_idx


def get_word_embedding(bert_output, start_idx, end_idx):
    return bert_output[:, start_idx:end_idx, :].mean(axis=1).detach().cpu().numpy()


def main():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertModel.from_pretrained(MODEL_PATH)
    
    embeddings = {}

    for word in [QUERY_WORD, POTENTIAL_SUBSTITUTE, WRONG_SUBSTITUTE,
                 WRONG_SUBSTITUTE2, VERY_WRONG_SUBSTITUTE,
                 VERY_WRONG_SUBSTITUTE2]:
        recipe_prepared = prepare_ingredients_for_prediction(
            RECIPE.replace(QUERY_WORD, word))
        tokenized_input = tokenizer(
            recipe_prepared,
            return_tensors="pt",
            is_split_into_words=True
        )

        tokenized_word = tokenizer(
            word,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        output = model(**tokenized_input)[0]
        start_idx, end_idx = locate_word(
            tokenized_word.input_ids.flatten(),
            tokenized_input.input_ids.flatten()
        )
        
        embeddings[word] = get_word_embedding(output, start_idx, end_idx)

    cosines = {}
    for key in embeddings.keys():
        cosines[key] = cosine_similarity(embeddings[QUERY_WORD], embeddings[key])

    print(cosines)


if __name__ == "__main__":
    main()
