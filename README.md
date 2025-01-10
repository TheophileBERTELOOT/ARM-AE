# ARM-AE
This is the code of the experiences described in the article "Association rules mining with auto-encoder" : https://link.springer.com/chapter/10.1007/978-3-031-77731-8_5 presented at IDEAL 2025 conference

# How to use
In order to use ARM-AE on your own data you first have to :

`poetry install`

then to use ARM-AE use : 

`poetry run ARMAE`

there are several possible parameters :

- --input-path or -ip is the input data path
- --armae-results-path' or -arp is the path for the results of ARM-AE
- --nb_epoch or -ne is the number of epoch of training for ARM-AE (default : 2)
- --batch_size or -bs is the batch size for ARM-AE training (default : 128)
- --learning_rate or -lr is the learning rate for ARM-AE traning (default : 10e-3)
- --likeness or -lk is the proportion of similar items in rule with the same consequent (default : 0.5)
- --number_of_rules or -nbor is the number of rule per consequent (default : 2)
- --nb_antecedents or -nba is the maximum number of antecedent in a rule (default : 2)
- --is-loaded-model or -ilm is a flag to know if you want to retrain the ARM-AE model again or not
- --model-path or -mp is the path of the saved models if you don't want to retrain the model

