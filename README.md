# RANLP_2021_Local_Vs_Global-Pruning

# Does local pruning offer task specific models to learn effectively ?

## Environment

* Python 3
* pytorch 1.3.0
* transformers 4.1.1
* tensorboardX 1.9
* Mostly codes are borrowed from https://github.com/lixin4ever/BERT-E2E-ABSA and https://github.com/howardhsu/DE-CNN

# Baselines or DE-CNN (For aspect-term extraction task)

Step 1: Go to directory either conv4 or conv6 or DE-CNN_pruning
```
cd Baselines/conv4
```

Step 2: Training models on SemEval laptop14 and restaurant16 dataset 

For laptop dataset:

Using local pruning
```
python script/train.py --domain laptop --pruning local
```

Using global pruning
```
python script/train.py --domain laptop --pruning global
```

For restaurant dataset:

Using local pruning
```
python script/train.py --domain restaurant --pruning local
```

Using global pruning
```
python script/train.py --domain restaurant --pruning global
```

Step 3: Testing on SemEval laptop14 and restaurant16 dataset 

For models trained on laptop dataset using local pruning:
```
python script/evaluation.py --domain laptop --pruning local
```

For models trained on laptop dataset using global pruning:
```
python script/evaluation.py --domain laptop --pruning local
```

For models trained on restaurant dataset using local pruning:
```
python script/evaluation.py --domain restaurant --pruning local
```

For models trained on restaurant dataset using global pruning:
```
python script/evaluation.py --domain restaurant --pruning local
```