# BERT

Simply run:
```commandline
python BERT.py --bert-type bert-base-cased
```
to run 5-fold cross-validation.
All tested BERT versions include:
* bert-base-cased
* bert-large-cased
* [FoodNER](https://github.com/ds4food/FoodNer/blob/master/FoodNER.ipynb) 
  checkpoint (the classification layer is excluded due to a different number 
  of predicted classes classes)

# LUKE

Simply run:
```commandline
accelerate launch luke_5fold.py --model_name_or_path studio-ousia/luke-base --task_name ner --max_length 128 --per_device_train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 30 --output_dir /tmp/ner/
```
Make sure the paths to luke_utils and src.luke are correct 
