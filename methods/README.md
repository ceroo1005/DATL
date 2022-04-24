# Semi-supervised comparison algorithm on PyTorch

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 


## Usage

1. Sample sampling by useing the following commands and arguments
```
python 1copyfile.py [--root root] [--save save]  [--smi-sample-rate 0.1] [--test-only]
```
| Argument | Default
| :--- | :----------
--root| /data
--save | /data/smisup
--smi-sample-rate | 0.1
--test-only | False

Use the parameter ```-test-only``` to sample on only one migration task (Office31 has six transfer tasks).

2. To run semi-supervised learning, you can use the following command
```
python 2smi.py [--root root] [--dataset dataset]  [--source source] [--target target] [--method method]
```
| Argument  | Default                
|:----------|:-----------------------
 --root    | /data/smisup           
 --dataset | Office31smiWD          
 --source  | webcam                 
 --target  | dslr                   
 --method  | ResNet

You can change ```--method``` parameters to decide which comparison methods to use.
```--method``` support the following options:'DAN'|'DDC'|'DANN'|'ResNet'.



