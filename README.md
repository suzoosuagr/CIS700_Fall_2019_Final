# CIS 700 Final NPI QUEUE and STACK
Neural Programmer-Interpreter Implementation in PyTorch. Here is the original paper: [Neural Programmer-Interpreters](https://arxiv.org/abs/1511.06279), by Reed and de Freitas.

We modified NPI to implement a differentiable queue and stack to solve the Maze and Reverse Polish Notation problems.

## Maze Problem
Please enter NPI_Maze folder at first. All files needed are in this folder.

### Environment
* numpy 1.18.1
* tensorboard 2.1.0
* tensorboardX 2.0
* torch 1.4.0+cpu or 1.0.1
* torchvision 0.5.0+cpu or 0.2.1

### Generate the maze data and solutions
See the NPI_Maze/Data/maze_gen.py.

### Training the model
We build 3 different training datasets and 3 different test datasets in NPI_Maze/Data/.
You can directly run .py files with prefix 'train'

```python
python train_n1_5_50.py
```
The example will train the model with network configuration 1 on the train_5_50 dataset.

### Evaluate the model
Here is a demo.
```python
python eval_n1_7_100.py
```

## Reverse Polish Problem

Please enter [reverse_polish](reverse_polish/) folder at first. All files needed are in this folder.

### Environment
* python 3.7.4
* numpy 1.18.4
* tensorboard 2.1.1
* tensorboardX 2.0
* torch 1.5.0
* torchvision 0.6.0

## Generate reverse polish trace
```bash
cd CIS700_Fall_2019_Final/reverse_polish/
python prepare_data.py
```
### Expressions Examples (maximum length 14)  
```text
C/F+(D*F+H)
E/B/H-(B+A/G)
H+F/C-(B+C/E)
G+D-F
H+D*E/F*H
E*D+B
E*A+D/G
```

### Example of trace
```text 
TRACE
Expression:     E/B/H-(B+A/G)..
Stack     :     ...............
-------------------------------
Rev Polish:     E..............
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     /..............
-------------------------------
Rev Polish:     E..............
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     /..............
-------------------------------
Rev Polish:     EB.............
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     /..............
-------------------------------
Rev Polish:     EB/............
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     ...............
-------------------------------
Rev Polish:     EB/............
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     /..............
-------------------------------
Rev Polish:     EB/............
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     /..............
-------------------------------
Rev Polish:     EB/H...........
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     /..............
-------------------------------
Rev Polish:     EB/H/..........
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     ...............
-------------------------------
Rev Polish:     EB/H/..........
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     -..............
-------------------------------
Rev Polish:     EB/H/..........
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     -(.............
-------------------------------
Rev Polish:     EB/H/..........
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     -(.............
-------------------------------
Rev Polish:     EB/H/B.........
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     -(+............
-------------------------------
Rev Polish:     EB/H/B.........
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     -(+............
-------------------------------
Rev Polish:     EB/H/BA........
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     -(+/...........
-------------------------------
Rev Polish:     EB/H/BA........
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     -(+/...........
-------------------------------
Rev Polish:     EB/H/BAG.......
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     -(+/...........
-------------------------------
Rev Polish:     EB/H/BAG/......
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     -(+............
-------------------------------
Rev Polish:     EB/H/BAG/......
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     -(+............
-------------------------------
Rev Polish:     EB/H/BAG/+.....
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     -(.............
-------------------------------
Rev Polish:     EB/H/BAG/+.....
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     -..............
-------------------------------
Rev Polish:     EB/H/BAG/+.....
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     -..............
-------------------------------
Rev Polish:     EB/H/BAG/+-....
###############################
Expression:     E/B/H-(B+A/G)..
Stack     :     ...............
-------------------------------
Rev Polish:     EB/H/BAG/+-....
###############################
__________
[
    (('REVPOLI', 2), [], False), (('PRECE', 3), [], False), (('WRITE', 1), [1, 5], False), (('MOV_PTR', 0), [2, 1], False), (('MOV_PTR', 0), [0, 1], False), (('PRECE', 3), [], False), (('MOV_PTR', 0), [1, 1], False), (('WRITE', 1), [1, 14], False), (('MOV_PTR', 0), [0, 1], False), (('PRECE', 3), [], False), (('WRITE', 1), [1, 2], False), (('MOV_PTR', 0), [2, 1], False), (('MOV_PTR', 0), [0, 1], False), (('PRECE', 3), [], False), (('WRITE', 1), [1, 14], False), (('MOV_PTR', 0), [2, 1], False), (('WRITE', 1), [1, 0], False), (('MOV_PTR', 0), [1, 0], False), (('MOV_PTR', 0), [1, 1], False), (('WRITE', 1), [1, 14], False), (('MOV_PTR', 0), [0, 1], False), (('PRECE', 3), [], False), (('WRITE', 1), [1, 8], False), (('MOV_PTR', 0), [2, 1], False), (('MOV_PTR', 0), [0, 1], False), (('PRECE', 3), [], False), (('WRITE', 1), [1, 14], False), (('MOV_PTR', 0), [2, 1], False), (('WRITE', 1), [1, 0], False), (('MOV_PTR', 0), [1, 0], False), (('MOV_PTR', 0), [1, 1], False), (('WRITE', 1), [1, 12], False), (('MOV_PTR', 0), [0, 1], False), (('PRECE', 3), [], False), (('MOV_PTR', 0), [1, 1], False), (('WRITE', 1), [1, 9], False), (('MOV_PTR', 0), [0, 1], False), (('PRECE', 3), [], False), (('WRITE', 1), [1, 2], False), (('MOV_PTR', 0), [2, 1], False), (('MOV_PTR', 0), [0, 1], False), (('PRECE', 3), [], False), (('MOV_PTR', 0), [1, 1], False), (('WRITE', 1), [1, 11], False), (('MOV_PTR', 0), [0, 1], False), (('PRECE', 3), [], False), (('WRITE', 1), [1, 1], False), (('MOV_PTR', 0), [2, 1], False), (('MOV_PTR', 0), [0, 1], False), (('PRECE', 3), [], False), (('MOV_PTR', 0), [1, 1], False), (('WRITE', 1), [1, 14], False), (('MOV_PTR', 0), [0, 1], False), (('PRECE', 3), [], False), (('WRITE', 1), [1, 7], False), (('MOV_PTR', 0), [2, 1], False), (('MOV_PTR', 0), [0, 1], False), (('PRECE', 3), [], False), (('WRITE', 1), [1, 14], False), (('MOV_PTR', 0), [2, 1], False), (('WRITE', 1), [1, 0], False), (('MOV_PTR', 0), [1, 0], False), (('WRITE', 1), [1, 11], False), (('MOV_PTR', 0), [2, 1], False), (('WRITE', 1), [1, 0], False), (('MOV_PTR', 0), [1, 0], False), (('WRITE', 1), [1, 0], False), (('MOV_PTR', 0), [1, 0], False), (('MOV_PTR', 0), [0, 1], False), (('WRITE', 1), [1, 12], False), (('MOV_PTR', 0), [2, 1], False), (('WRITE', 1), [1, 0], False), (('MOV_PTR', 0), [1, 0], False), (('MOV_PTR', 0), [1, 0], True)
]
```

## Train the model
There are several `train*.py` files under [reverse_polish](reverse_polish/) folder.
```bash
cd CIS700_Fall_2019_Final/reverse_polish/
python train_exp_len14_adam.py
```
## Test the model on different data
There are two example `.py` files. 
```bash
cd CIS700_Fall_2019_Final/reverse_polish/
python test_exp_len8_sgd_1en3_on_exp_len8.py
``` 
For more modification, please open [test_exp_len8_sgd_1en3_on_exp_len8.py](reverse_polish/test_exp_len8_sgd_1en3_on_exp_len8.py) and modify following:

1. Line 188: change `exp_dir` to loading specific trained weights.
2. Line 214: change `TEST_DATA_PATH`  to loading different data. `test.pik` is the data with maximum-length 14, and `test_8.pik` is the data with maximum-length 8. 
