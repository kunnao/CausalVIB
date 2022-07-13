# CEVIB

Causal effect estimation using variational information bottleneck

This code is written by Mingjun Zhong and Zhenyu Lu:

References: Causal effect estimation using variational information bottleneck


**Requirements**

0. This software was tested on windows 10
1. Create your virtual environment Python 3.5-3.8
2. Install Tensorflow = 2.5.0

   * Follow official instruction on https://www.tensorflow.org/install/

   * Remember a GPU support is highly recommended for training
3. Clone this repository

For instance, the environments we used are listed in the file `requirement.txt` - 
you could find all the packages there. If you use `conda` or `pip`, 
you may type `pip install -r requirements.txt` to set up the environment.
    

# How to use the code and examples

With this project you will be able to use the Sequence to Point network. You can prepare the dataset from the
most common in NILM, train the network and test it. Target appliances taken into account are kettle, microwave, fridge, dish washer and
washing machine.
Directory tree:

``` bash
CausalVIB/
│--.gitignore
│--data_feeder.py
│--detail_train.py
│--evalution.py
│--model_structure.py
│--nilm_metric.py
│--plot.py
│--plot_res.py
│--read.py
│--README.md
│--remove_space.py
│--requirements.txt
│--seq2point_train.py
│--test_on_running.py
│--train_main.py
│--utils.py
│--val_main.py
│--Variational_information_bottleneck_for_causal_inference (1).pdf
│
└─semi_parametric_estimation
   │--ate.py
   │--att.py
   │--helpers.py
```

python train_main.py --network_type dragonvib --dataset acic --targeted_regularization 0 --batch_size 128 --replication 5

## **Create ACIC, IHDP, TWINS or SIMU_BIAS dataset**


### ACIC

Download the ACIC data from the original website ([IBM-Causal-Inference-Benchmarking-Framework/data/LBIDD at master · IBM-HRL-MLHLS/IBM-Causal-Inference-Benchmarking-Framework · GitHub](https://github.com/IBM-HRL-MLHLS/IBM-Causal-Inference-Benchmarking-Framework/tree/master/data/LBIDD)). 
We tested our model on 'scaling' dataset.


### IHDP

Download the IHDP raw data from the original website (https://github.com/OsierYi/SITE/tree/master/data) or use the data we have preprocessed in ./data/ihdp.npz.


### TWINS

Download the TWINS data from the original website (https://github.com/jsyoon0823/GANITE/tree/master/data)  or use the data we have preprocessed in ./data/twins.npz.



**I will write instructions how to use the code with more details. Currently, you just run train_main.py and test_main.py. Do remember to choose your parameters in these two files correspondingly.**

To train the model, just run `python train_main.py` or in IDE environment, e.g., Spyder, run train_main.py

For example:

```python train_main.py --network_type causalvib --dataset ihdp --batch_size 128 --replication 50```

To test the model, just run `python val_main.py` or in IDE environment, e.g., Spyder, run val_main.py

Any questions, please write email to me: zhenyulu98@foxmail.com