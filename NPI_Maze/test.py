from __future__ import print_function
import argparse
import os
import torch
import numpy as np
from Model.npi import NPI
from Model.maze_core import MazeCore
import pickle
from Env.maze import Maze_Env, CONFIG
import time
from tensorboardX import SummaryWriter
import random

n_iter = 0

def run_epoch(npi, mode, cur_data, writer):
    random.shuffle(cur_data)
    global n_iter
    npi.eval()

    epoch_def_loss = 0.0
    epoch_total_loss = 0.0
    epoch_pro_accs = 0.0
    epoch_ter_accs = 0.0
    epoch_step = 0
    start_time = time.time()
    for maze_idx in range(len(cur_data)):
        start, end, maze, trace = cur_data[maze_idx]

        maze_env = Maze_Env(start, end, maze)

        x, y = trace[:-1], trace[1:]

        # step_def_loss, step_arg_loss, term_acc, prog_acc, = 0.0, 0.0, 0.0, 0.0
        # arg0_acc, arg1_acc, arg2_acc, num_args = 0.0, 0.0, 0.0, 0
        step_def_loss = 0.0
        step_total_loss = 0.0
        pro_accs = 0.0
        ter_accs = 0.0

        for trace_idx in range(len(x)):
            (pro_in_name, pro_in_id), arg_in, ter_in = x[trace_idx]
            (pro_out_name, pro_out_id), arg_out, ter_out = y[trace_idx]

            maze_env.execute(pro_in_id, arg_in)
            env_ft = maze_env.encode_env()
            env_ft = torch.from_numpy(env_ft).view(1, -1)

            arg_in_ft = maze_env.encode_args(arg_in)
            arg_in_ft = torch.from_numpy(arg_in_ft).view(1, -1)
            arg_out_ft = maze_env.encode_args(arg_out)
            arg_out_ft = torch.from_numpy(arg_out_ft).view(1, -1)

            pro_in_ft = np.array([pro_in_id])
            pro_in_ft = torch.from_numpy(pro_in_ft).view(1, -1)
            pro_out_ft = np.array([pro_out_id])
            pro_out_ft = torch.from_numpy(pro_out_ft).view(-1)

            ter_out_ft = [1] if ter_out else [0]
            ter_out_ft = np.array(ter_out_ft)
            ter_out_ft = torch.from_numpy(ter_out_ft).view(-1)

            if cuda_flag:
                env_ft = env_ft.cuda()
                arg_in_ft = arg_in_ft.cuda()
                arg_out_ft = arg_out_ft.cuda()
                pro_in_ft = pro_in_ft.cuda()
                pro_out_ft = pro_out_ft.cuda()
                ter_out_ft = ter_out_ft.cuda()

            pro_pred, arg_pred, ter_pred = npi(env_ft, arg_in_ft, pro_in_ft)
            pred = (pro_pred, arg_pred, ter_pred)
            gt = (pro_out_ft, arg_out_ft, ter_out_ft)
            default_loss, total_loss = npi.cal_loss(pred, gt)

            pro_acc, ter_acc = npi.cal_metrics(pred, gt)
            pro_accs += pro_acc
            ter_accs += ter_acc

            step_def_loss += default_loss.item()
            step_total_loss += total_loss.item()

        if maze_idx % 10 == 0:
            print("Mode {0:s} Maze idx {1:03d} Default Step Loss {2:05f}, " \
                  "Total Step Loss {3:05f}, Term Acc: {4:03f}, Prog Acc: {5:03f}" \
                  .format(mode, maze_idx, step_def_loss / len(x), step_total_loss / len(x), ter_accs / len(x),
                          pro_accs / len(x)))

        writer.add_scalar(mode + '/def_loss', step_def_loss / len(x), n_iter)
        writer.add_scalar(mode + '/total_loss', step_total_loss / len(x), n_iter)
        writer.add_scalar(mode + '/pro_accs', pro_accs / len(x), n_iter)
        writer.add_scalar(mode + '/ter_accs', ter_accs / len(x), n_iter)
        n_iter += 1

        epoch_def_loss += step_def_loss
        epoch_total_loss += step_total_loss
        epoch_pro_accs += pro_accs
        epoch_ter_accs += ter_accs
        epoch_step += len(x)

    end_time = time.time()
    epoch_time = end_time - start_time
    print("Mode: {0:s} For whole Dataset, Time Consum {1:05f} Default Step Loss {1:05f}, " \
          "Total Step Loss {3:05f}, Term Acc: {4:03f}, Prog Acc: {5:03f}"
          .format(mode, epoch_time, epoch_def_loss / epoch_step,
                  epoch_total_loss / epoch_step, epoch_ter_accs / epoch_step,
                  epoch_pro_accs / epoch_step))
    print('===============================')
    return (epoch_def_loss / epoch_step, epoch_total_loss / epoch_step,
            epoch_ter_accs / epoch_step, epoch_pro_accs / epoch_step)

def print_net(network):
    print('---------- Networks initialized -------------')
    num_params = 0
    for param in network.parameters():
        num_params += param.numel()
    print(network)
    print('NPI: Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')

if __name__ == "__main__":
    start_epoch = 1
    max_num_epochs = 30
    exp_dir = os.path.join('tfboard', 'n1_5_100_test')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    writer = SummaryWriter(exp_dir)

    TRAIN_DATA_PATH = 'Data/train_5_100.pik'
    with open(TRAIN_DATA_PATH, 'rb', ) as f:
        train_data = pickle.load(f)

    TEST_SAME_DATA_PATH = 'Data/test_5_100.pik'
    with open(TEST_SAME_DATA_PATH, 'rb', ) as f:
        test_same_data = pickle.load(f)

    TEST_DIF_DATA_PATH = 'Data/test_10_100.pik'
    with open(TEST_DIF_DATA_PATH, 'rb', ) as f:
        test_dif_data = pickle.load(f)

    if torch.cuda.is_available():
        cuda_flag = True
        CUDA_VISIBLE_DEVICES = '0'
        print('Current GPU index: ' + CUDA_VISIBLE_DEVICES)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    else:
        cuda_flag = False

    maze_core = MazeCore()
    npi = NPI(maze_core, CONFIG)

    save_path = os.path.join(exp_dir, 'npi_last.pth')
    npi.save_network(save_path, cuda_flag)
    if cuda_flag:
        npi = npi.cuda()

    if cuda_flag:
        npi = npi.cuda()

    print_net(npi)

    mode = 'test_same'
    cur_same_results = run_epoch(npi, mode, test_same_data, writer)

    same_def_loss, same_total_loss, same_ter_accs, same_pro_accs = cur_same_results

    mode = 'test_dif'
    cur_diff_results = run_epoch(npi, mode, test_dif_data, writer)
    diff_def_loss, diff_total_loss, diff_ter_accs, dif_pro_accs = cur_diff_results






