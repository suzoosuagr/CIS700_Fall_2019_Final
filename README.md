# CIS 700 Final NPI QUEUE and STACK

## Maze Problem
Please NPI_Maze folder at first. All files needed are in this folder.

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
