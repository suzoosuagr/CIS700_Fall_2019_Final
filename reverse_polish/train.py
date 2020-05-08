import os
import torch
import numpy as np
import pickle

from model.npi import NPI
from model.revpolish_core import RevPolishCore
from tasks.reverse_polish.config import ScratchPad, CONFIG

import time
from tensorboardX import SummaryWriter
import random

torch.autograd.set_detect_anomaly(True)

# FIX SEEDS. 
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)

train_n_iter = 0
same_n_iter = 0
diff_n_iter = 0

def run_epoch(npi, mode, cur_data, writer):
    random.shuffle(cur_data)
    global train_n_iter
    global same_n_iter
    global diff_n_iter

    if mode != 'train':
        npi.eval()
    else:
        npi.train()

    epoch_def_loss = 0.0
    epoch_total_loss = 0.0
    epoch_pro_accs = 0.0
    epoch_ter_accs = 0.0
    epoch_step = 0
    start_time = time.time()
    for idx in range(len(cur_data)):
        exp, trace = cur_data[idx]

        revpolish_env = ScratchPad(exp)

        x, y = trace[:-1], trace[1:]

        hidden = torch.zeros(2, 1, 256, requires_grad=True)
        cell   = torch.zeros(2, 1, 256, requires_grad=True)
        if cuda_flag:
            hidden = hidden.cuda()
            cell   = cell.cuda()
        # npi.reset_state(1)
        # step_def_loss, step_arg_loss, term_acc, prog_acc, = 0.0, 0.0, 0.0, 0.0
        # arg0_acc, arg1_acc, arg2_acc, num_args = 0.0, 0.0, 0.0, 0
        step_def_loss = 0.0
        step_total_loss = 0.0
        pro_accs = 0.0
        ter_accs = 0.0

        for trace_idx in range(len(x)):
            (pro_in_name, pro_in_id), arg_in, ter_in = x[trace_idx]
            (pro_out_name, pro_out_id), arg_out, ter_out = y[trace_idx]

            revpolish_env.execute(pro_in_id, arg_in)
            env_ft = revpolish_env.get_env()
            env_ft = torch.from_numpy(env_ft).view(1, -1)

            arg_in_ft = revpolish_env.encode_args(arg_in)
            arg_in_ft = torch.from_numpy(arg_in_ft).view(1, -1)
            arg_out_ft = revpolish_env.encode_args(arg_out)
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
                
            # if trace_idx == 0:
            #     # first step:
            #     pro_pred, arg_pred, ter_pred, hidden, cell = npi(env_ft, arg_in_ft, pro_in_ft, hidden=None, cell=None)
            # else:
            pro_pred, arg_pred, ter_pred, _, _ = npi(env_ft, arg_in_ft, pro_in_ft, hidden, cell)
            pred = (pro_pred, arg_pred, ter_pred)
            gt = (pro_out_ft, arg_out_ft, ter_out_ft)
            default_loss, total_loss = npi.cal_loss(pred, gt)

            pro_acc, ter_acc = npi.cal_metrics(pred, gt)
            pro_accs += pro_acc
            ter_accs += ter_acc

            if mode == 'train':
                # arg is not blank
                if pro_out_id == 0 or pro_out_id == 3 \
                        or pro_out_id == 4 or pro_out_id == 7:
                    optimizer.zero_grad()
                    total_loss.backward(retain_graph=True)
                    optimizer.step()
                else:  # ter_loss and pro_loss
                    optimizer.zero_grad()
                    default_loss.backward(retain_graph=True)
                    optimizer.step()

            step_def_loss += default_loss.item()
            step_total_loss += total_loss.item()

        if idx % 10 == 0:
            print("Epoch {0:02d} Maze idx {1:03d} Default Step Loss {2:05f}, " \
                  "Total Step Loss {3:05f}, Term Acc: {4:03f}, Prog Acc: {5:03f}" \
                  .format(curr_epoch, idx, step_def_loss / len(x), step_total_loss / len(x), ter_accs / len(x),
                          pro_accs / len(x)))

        if mode == 'train':
            writer.add_scalar(mode + '/def_loss', step_def_loss / len(x), train_n_iter)
            writer.add_scalar(mode + '/total_loss', step_total_loss / len(x), train_n_iter)
            writer.add_scalar(mode + '/pro_accs', pro_accs / len(x), train_n_iter)
            writer.add_scalar(mode + '/ter_accs', ter_accs / len(x), train_n_iter)
            train_n_iter += 1
        elif mode == 'test_same':
            writer.add_scalar(mode + '/def_loss', step_def_loss / len(x), same_n_iter)
            writer.add_scalar(mode + '/total_loss', step_total_loss / len(x), same_n_iter)
            writer.add_scalar(mode + '/pro_accs', pro_accs / len(x), same_n_iter)
            writer.add_scalar(mode + '/ter_accs', ter_accs / len(x), same_n_iter)
            same_n_iter += 1
        elif mode == 'test_dif':
            writer.add_scalar(mode + '/def_loss', step_def_loss / len(x), diff_n_iter)
            writer.add_scalar(mode + '/total_loss', step_total_loss / len(x), diff_n_iter)
            writer.add_scalar(mode + '/pro_accs', pro_accs / len(x), diff_n_iter)
            writer.add_scalar(mode + '/ter_accs', ter_accs / len(x), diff_n_iter)
            diff_n_iter += 1

        epoch_def_loss += step_def_loss
        epoch_total_loss += step_total_loss
        epoch_pro_accs += pro_accs
        epoch_ter_accs += ter_accs
        epoch_step += len(x)

    end_time = time.time()
    epoch_time = end_time - start_time
    print("Mode: {0:s} For whole Epoch {1:02d}, Time Consum {2:05f} Default Step Loss {3:05f}, " \
          "Total Step Loss {4:05f}, Term Acc: {5:03f}, Prog Acc: {6:03f}"
          .format(mode, curr_epoch, epoch_time, epoch_def_loss / epoch_step,
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
    max_num_epochs = 20
    exp_dir = os.path.join('tfboard', 'exp_len15')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    writer = SummaryWriter(exp_dir)

    # according to the results of test same
    Best_results = dict()
    Best_results['def_loss'] = 1000000.0
    Best_results['epoch_def_loss'] = -1
    Best_results['total_loss'] = 1000000.0
    Best_results['epoch_total_loss'] = -1
    Best_results['ter_accs'] = 0.0
    Best_results['epoch_ter_accs'] = -1
    Best_results['pro_accs'] = 0.0
    Best_results['epoch_pro_accs'] = -1

    TRAIN_DATA_PATH = 'tasks/reverse_polish/data/train.pik'
    with open(TRAIN_DATA_PATH, 'rb', ) as f:
        train_data = pickle.load(f)

    TEST_SAME_DATA_PATH = 'tasks/reverse_polish/data/eval.pik'
    with open(TEST_SAME_DATA_PATH, 'rb', ) as f:
        test_same_data = pickle.load(f)

    TEST_DIF_DATA_PATH = 'tasks/reverse_polish/data/test.pik'
    with open(TEST_DIF_DATA_PATH, 'rb', ) as f:
        test_dif_data = pickle.load(f)

    CUDA_VISIBLE_DEVICES = '0'
    print('Current GPU index: ' + CUDA_VISIBLE_DEVICES)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    if torch.cuda.is_available():
        cuda_flag = True
    else:
        cuda_flag = False

    revpolish_core = RevPolishCore()
    npi = NPI(revpolish_core, CONFIG)
    if cuda_flag:
        npi = npi.cuda()

    print_net(npi)

    parameters = filter(lambda p: p.requires_grad,
                        npi.parameters())
    # optimizer = torch.optim.SGD(parameters,
    #                             lr=0.0001,
    #                             momentum=0.9,
    #                             nesterov=True,
    #                             weight_decay=5e-4)
    optimizer = torch.optim.Adam(parameters, lr=0.0001, betas=(0.5, 0.999))
    lr_schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=max_num_epochs)

    for curr_epoch in range(start_epoch, max_num_epochs + 1):
        mode = 'train'
        run_epoch(npi, mode, train_data, writer)

        lr_schedulers.step(curr_epoch)
        if curr_epoch % 2 == 0:
            model_name = 'npi_' + str(curr_epoch) + '.pth'
            save_path = os.path.join(exp_dir, model_name)
            npi.save_network(save_path, cuda_flag)
            if cuda_flag:
                npi = npi.cuda()

        mode = 'test_same'
        cur_results = run_epoch(npi, mode, test_same_data, writer)

        def_loss, total_loss, ter_accs, pro_accs = cur_results

        if def_loss < Best_results['def_loss']:
            Best_results['def_loss'] = def_loss
            Best_results['epoch_def_loss'] = curr_epoch
        if total_loss < Best_results['total_loss']:
            Best_results['total_loss'] = total_loss
            Best_results['epoch_total_loss'] = curr_epoch
        if ter_accs > Best_results['ter_accs']:
            Best_results['ter_accs'] = ter_accs
            Best_results['epoch_ter_accs'] = curr_epoch
        if pro_accs > Best_results['pro_accs']:
            Best_results['pro_accs'] = pro_accs
            Best_results['epoch_pro_accs'] = curr_epoch

        # mode = 'train'
        # run_epoch(npi, mode, test_dif_data, writer)

    save_path = os.path.join(exp_dir, 'npi_last.pth')
    npi.save_network(save_path, cuda_flag)
    if cuda_flag:
        npi = npi.cuda()

    for key, val in Best_results.items():
        if key.find('epoch') != -1:
            print('Best %s for test same at %d epoch' % (key, val))
        else:
            print('Best %s for test same: %f' % (key, val))

