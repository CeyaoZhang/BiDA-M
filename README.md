The main.py works under pytorch v1.9.0 + CUDA 10.2 + cuDNN v7.6.5.
Some default settings for Huber and Tukey (biweight) loss functions, where the subscript in $c$ represents asymptotic relative efficiency (ARE) under the Gaussian noise case.

|loss| $c_{.85}$ | $c_{.90}$ | $c_{.95}$ | $c_{.98}$ | $c_{.99}$ |
|:---:|---|---|---|---|---|
| Huber | 0.7317 | 0.9818 | 1.345 | 1.7459 | 2.0102 |
| Tukey (biweight) | 3.4437 | 3.8827 | 4.685 | 5.9207 | 7.0414 | 

We list the exec codes as follows. For more details, check the *run.sh*, and you can run it by `sh run.sh`

# [linear01] 

## only 
```
python main.py --func linear01 --data_seed 555 --train_mode sigmas-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0
python main.py --func linear01 --data_seed 555 --train_mode epsilons-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0

python main.py --func linear01 --data_seed 555 --train_mode sigmas-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0 --bmodel Linear
python main.py --func linear01 --data_seed 555 --train_mode epsilons-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0 --bmodel Linear


```


## BiDA-Huber
 ```python
 - python main.py --func linear01 --data_seed 666 --train_mode regular --epochs 1000 --noise_type Laplace 
 - python main.py --func linear01 --data_seed 666 --train_mode sigmas-seeds --epochs 1000 --noise_type Laplace 
 - python main.py --func linear01 --data_seed 666 --train_mode epsilons-seeds --epochs 1000 --noise_type Laplace 

 ```

 - 15mins for a seed with method2+method3
 - [500 epochs] 10min for a seed, 5h for 30 seeds with method1+method3
 - [800 epochs] 12mins for a seed, 10h for 50 seeds

## BiDA-Tukey
 ```python
 - python main.py --func linear01 --data_seed 6 --train_mode regular --epochs 1000 --noise_type Laplace --adap_loss Tukey --hypara 4.685 
 - python main.py --func linear01 --data_seed 6 --train_mode sigmas --epochs 1000 --noise_type Laplace --adap_loss Tukey --hypara 4.685
 - python main.py --func linear01 --data_seed 6 --train_mode epsilons --epochs 1000 --noise_type Laplace --adap_loss Tukey --hypara 4.685 
 - python main.py --func linear01 --data_seed 6 --train_mode sigmas-seeds --epochs 1000 --noise_type Laplace --adap_loss Tukey --hypara 4.685 
 - python main.py --func linear01 --data_seed 6 --train_mode epsilons-seeds --epochs 1000 --noise_type Laplace --adap_loss Tukey --hypara 4.685 

 - python main.py --func linear01 --data_seed 6 --train_mode sigmas --epochs 1000 --noise_type Lognormal --adap_loss Tukey --hypara 4.685
 
 - python main.py --func linear01 --data_seed 6 --train_mode sigmas-seeds --epochs 2000 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 
 
 - nohup python -u main.py --func linear01 --data_seed 6 --train_mode sigmas-seeds --epochs 2000 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 >> my.log 2>&1 &

 - python main.py --func linear01 --data_seed 6 --train_mode epsilons-seeds --epochs 1000 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 
 ```

- [1000 epochs] 25mins for a seed with method1+method3, 
- [1000 epochs] 17~20h for 50 seeds with method1+method3, 2 exec run simultaneously
- [1000 epochs] 27h for 50 seeds with method1+method3, 4 exec run simultaneously
- 50 seeds are [50, 98, 54, 6, 34, 66, 63, 52, 39, 62, 46, 75, 28, 65, 18, 37, 85, 13, 80, 33, 69, 78, 19, 40, 82, 10, 43, 61, 88, 89, 56, 41, 27, 90, 57, 95, 4, 92, 59, 36, 72, 1, 96, 47, 97, 26, 70, 51, 73, 68]


# [sin01] 

```
python main.py --func sin01 --data_seed 555 --train_mode regular --epochs 3000 --noise_type Lognormal
python main.py --func sin01 --data_seed 40 --train_mode sigmas-seeds --epochs 3000
python main.py --func sin01 --data_seed 555 --train_mode sigmas-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal
python main.py --func sin01 --data_seed 555 --train_mode epsilons-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal


```

- [500 epochs] 8mins for a seed, 4h for 30 seeds with method3=meta+meta2,
- [750 epochs] 20mins for one seed with 5 sigmas and method2+method3, 
- [1500 epochs] 12 mins for a seed in a sigma and 5 methods, 
- [2000 epochs] almost 45mins for a seed, 8h for 10 seeds with method2+method3
- [2000 epochs] 9h for 10 seeds with method2+method3, [59, 75, 68, 5, 32, 37, 86, 82, 27, 17]
- [3000 epochs] 52h for 15 seeds with method2+method3, 7Huber+3others [50, 98, 54, 6, 34, 66, 63, 52, 39, 62, 46, 75, 28, 65, 18]
- [3000 epochs] 50h for 50 seeds with [25, 36, 21, 17, 72, 91, 14, 20, 33, 53, 99, 71, 37, 77, 58, 3, 96, 75, 5, 63, 57, 6, 93, 16, 11, 42, 54, 94, 74, 19]


# [poly5] 
- [360 epochs] 11mins for one seeds, 6h for 30 seeds, method2+method3


# [tan01]
```
python train.py --func tan01 --data_seed 666 --mode normal --epochs 1000 --noise_type Lognormal


python main.py --func tan01 --data_seed 6666 --train_mode regular --epochs 2000 --noise_type Lognormal

python main.py --func tan01 --data_seed 6666 --train_mode regular --epochs 2000 --noise_type Lognormal --outer_loss Huber

python main.py --func tan01 --data_seed 666 --train_mode regular --epochs 2000 --noise_type Lognormal --noisy_val 1 --outer_loss Huber

python main.py --func tan01 --data_seed 666 --train_mode regular --epochs 2000 --noise_type Lognormal --noisy_val 1 --outer_loss MAE


python main.py --func tan01 --data_seed 66 --train_mode sigmas-seeds --epochs 1500 --num_seeds 15 --noise_type Lognormal
python main.py --func tan01 --data_seed 66 --train_mode epsilons-seeds --epochs 1500 --num_seeds 15 --noise_type Lognormal


- 28h for 15 seeds [50, 98, 54, 6, 34, 66, 63, 52, 39, 62, 46, 75, 28, 65, 18]
```

# [log01]
```
python main.py --func log01 --data_seed 66 --train_mode regular --epochs 1500 --noise_type Lognormal


--

python main.py --func log01 --data_seed 66 --train_mode sigmas-seeds --epochs 1500 --num_seeds 15 --noise_type Lognormal
python main.py --func log01 --data_seed 66 --train_mode epsilons-seeds --epochs 1500 --num_seeds 15 --noise_type Lognormal

python main.py --func log01 --data_seed 6 --train_mode sigmas-seeds --epochs 1500 --num_seeds 15 --noise_type Lognormal --noisy_val 1 --outer_loss Huber

- 28h for 15 seeds [50, 98, 54, 6, 34, 66, 63, 52, 39, 62, 46, 75, 28, 65, 18]


```


# some new baseline settings

## Level 1: use linear model as the backbone for BiDA-M
```
python main.py --func linear01 --data_seed 666 --train_mode sigmas-seeds --epochs 1000 --noise_type Laplace  --adap_loss Huber --model Linear 
python main.py --func linear01 --data_seed 666 --train_mode sigmas-seeds --epochs 1000 --noise_type Lognormal --adap_loss Huber --model Linear 

python main.py --func linear01 --data_seed 6 --train_mode sigmas-seeds --epochs 1000 --noise_type Laplace --adap_loss Tukey --hypara 4.685 --model Linear 
python main.py --func linear01 --data_seed 6 --train_mode sigmas-seeds --epochs 1000 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 --model Linear 
```

## Level 2: add validation data to the training data for the baseline methods
```
python main.py --func linear01 --data_seed 666 --train_mode sigmas-seeds --epochs 1000 --noise_type Laplace  --adap_loss Huber --model NN --train_add_val 1
python main.py --func linear01 --data_seed 666 --train_mode sigmas-seeds --epochs 1000 --noise_type Lognormal --adap_loss Huber --model NN --train_add_val 1

python main.py --func linear01 --data_seed 6 --train_mode sigmas-seeds --epochs 1000 --noise_type Laplace --adap_loss Tukey --hypara 4.685 --model NN --train_add_val 1
python main.py --func linear01 --data_seed 6 --train_mode sigmas-seeds --epochs 1000 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 --model NN --train_add_val 1
```

## Level 3: the validation data also contains the noise`



