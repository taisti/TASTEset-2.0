# BERT

Simply run:
```commandline
python BERT.py --bert-type bert-large-cased
```
to run 5-fold cross-validation.
All tested BERT versions include:
* bert-base-uncased
* bert-base-cased
* bert-large-cased
* [FoodNER](https://github.com/ds4food/FoodNer/blob/master/FoodNER.ipynb) 
  checkpoint (the classification layer is excluded due to a different number 
  of predicted classes classes)
* [BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)

# LUKE

Simply run:
```commandline
accelerate launch luke_5fold.py --model_name_or_path studio-ousia/luke-base --task_name ner --max_length 128 --per_device_train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 30 --output_dir /tmp/ner/
```
alternatively
```commandline
python luke_5fold.py --model_name_or_path studio-ousia/luke-base --task_name ner --max_length 128 --per_device_train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 30 --output_dir /tmp/ner/
```
