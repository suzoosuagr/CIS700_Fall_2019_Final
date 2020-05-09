import os
import numpy as np
import random 
import datetime
import torch
import pickle
import time

from tensorboardX import SummaryWriter
from model.npi import NPI, NPI_LOSS
from model.revpolish_core import RevPolishCore
from tasks.reverse_polish.config import CONFIG, ScratchPad


# fix random seed.
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)

# device
if not torch.cuda.is_available():
    device = 'cpu'
else:
    device = 'cuda'
device = torch.device(device)


train_n_iter = 0
same_n_iter = 0
diff_n_iter = 0


def run_epoch(model, mode, cur_data, writer):
    global train_n_iter
    global same_n_iter
    global diff_n_iter
    if mode == 'train':
        random.shuffle(cur_data)
        model.train()
    else:
        model.eval()

    epoch_def_loss = 0.0
    epoch_total_loss = 0.0
    epoch_pro_accs = 0.0
    epoch_ter_accs = 0.0
    epoch_step = 0
    start_time = time.time()

    criterion = NPI_LOSS()

    for idx in range(len(cur_data)):
        exp, trace = cur_data[idx]
        Pad = ScratchPad(exp)
        x, y = trace[:-1], trace[1:]
        h0 = torch.zeros((2, 1, 256)).to(device)
        
        step_def_loss = 0.0
        step_total_loss = 0.0
        pro_accs = 0.0
        ter_accs = 0.0

        for trace_idx in range(len(x)):
            (pro_in_name, pro_in_id), arg_in, ter_in = x[trace_idx]
            (pro_out_name, pro_out_id), arg_out, ter_out = y[trace_idx]

            Pad.execute(pro_in_id, arg_in)
            env_ft = Pad.get_env()
            env_ft = torch.from_numpy(env_ft).view(1, -1)  # the value of exp

            arg_in_ft = Pad.encode_args(arg_in)
            arg_in_ft = torch.from_numpy(arg_in_ft).view(1, -1)
            arg_out_ft = Pad.encode_args(arg_out)
            arg_out_ft = torch.from_numpy(arg_out_ft).view(1, -1)
            # arg_out_ft = np.array(arg_out)
            # arg_out_ft = torch.from_numpy(arg_out_ft).view(1, -1)


            pro_in_ft = np.array([pro_in_id])
            pro_in_ft = torch.from_numpy(pro_in_ft).view(1, -1)
            pro_out_ft = np.array([pro_out_id])
            pro_out_ft = torch.from_numpy(pro_out_ft).view(-1)

            ter_out_ft = [1] if ter_out else [0]
            ter_out_ft = np.array(ter_out_ft)
            ter_out_ft = torch.from_numpy(ter_out_ft).view(-1)

            arg_in_ft = arg_in_ft.to(device)
            arg_out_ft = arg_out_ft.to(device)
            pro_in_ft = pro_in_ft.to(device)
            pro_out_ft = pro_out_ft.to(device)
            ter_out_ft = ter_out_ft.to(device)
            env_ft = env_ft.to(device)

            initial = (trace_idx==0)
            pred, _ = npi(env_ft, arg_in_ft, pro_in_ft, h0, initial)
            gt = (ter_out_ft, pro_out_ft, arg_out_ft)

            default_loss, total_loss = criterion(pred, gt)
            pro_acc, ter_acc = criterion.metric(pred, gt)
            pro_accs += pro_acc
            ter_accs += ter_acc

            if mode == 'train':
                if pro_out_id == 0 or pro_out_id == 1 or pro_out_id == 4:
                    optimizer.zero_grad()
                    total_loss.backward(retain_graph=True)
                    optimizer.step()
                else:
                    optimizer.zero_grad()
                    default_loss.backward(retain_graph=True)
                    optimizer.step()

                step_def_loss += default_loss.item()
                step_total_loss += total_loss.item()

        if idx % 10 == 0:
            
            print("Epoch {0:02d} idx {1:03d} Default Step Loss {2:05f}, " \
                  "Total Step Loss {3:05f}, Term Acc: {4:03f}, Prog Acc: {5:03f}" \
                  .format(epoch, idx, step_def_loss / len(x), step_total_loss / len(x), ter_accs / len(x),
                          pro_accs / len(x)))

        if mode == 'train':
            writer.add_scalar(mode + '/def_loss', step_def_loss / len(x), train_n_iter)
            writer.add_scalar(mode + '/total_loss', step_total_loss / len(x), train_n_iter)
            writer.add_scalar(mode + '/pro_accs', pro_accs / len(x), train_n_iter)
            writer.add_scalar(mode + '/ter_accs', ter_accs / len(x), train_n_iter)
            train_n_iter += 1
        elif mode == 'eval':
            writer.add_scalar(mode + '/def_loss', step_def_loss / len(x), same_n_iter)
            writer.add_scalar(mode + '/total_loss', step_total_loss / len(x), same_n_iter)
            writer.add_scalar(mode + '/pro_accs', pro_accs / len(x), same_n_iter)
            writer.add_scalar(mode + '/ter_accs', ter_accs / len(x), same_n_iter)
            same_n_iter += 1
        elif mode == 'test':
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
          .format(mode, epoch, epoch_time, epoch_def_loss / epoch_step,
                  epoch_total_loss / epoch_step, epoch_ter_accs / epoch_step,
                  epoch_pro_accs / epoch_step))
    print('===============================')
    return (epoch_def_loss / epoch_step, epoch_total_loss / epoch_step,
            epoch_ter_accs / epoch_step, epoch_pro_accs / epoch_step)

def test_epoch(model, metric, data):
    


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
    max_num_epochs = 6
    exp_dir = os.path.join('tfboard', 'exp_len8_sgd_1en3')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    writer = SummaryWriter(exp_dir)

    Best_results = dict()
    Best_results['def_loss'] = 1000000.0
    Best_results['epoch_def_loss'] = -1
    Best_results['total_loss'] = 1000000.0
    Best_results['epoch_total_loss'] = -1
    Best_results['ter_accs'] = 0.0
    Best_results['epoch_ter_accs'] = -1
    Best_results['pro_accs'] = 0.0
    Best_results['epoch_pro_accs'] = -1


    TRAIN_DATA_PATH = 'tasks/reverse_polish/data/train_8.pik'
    with open(TRAIN_DATA_PATH, 'rb', ) as f:
        train_data = pickle.load(f)

    EVAL_DATA_PATH = 'tasks/reverse_polish/data/eval_8.pik'
    with open(EVAL_DATA_PATH, 'rb', ) as f:
        eval_data = pickle.load(f)

    TEST_DATA_PATH = 'tasks/reverse_polish/data/test_8.pik'
    with open(TEST_DATA_PATH, 'rb', ) as f:
        test_data = pickle.load(f)

    func_core = RevPolishCore().to(device)
    npi       = NPI(func_core, CONFIG).to(device)

    print_net(npi)

    # optimizer = torch.optim.Adam(npi.parameters(), lr=1e-4)
    optimizer = torch.optim.SGD(npi.parameters(), lr=1e-3)
    lr_schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_num_epochs)

    for epoch in range(start_epoch, max_num_epochs+1):
        
        run_epoch(npi, 'train', train_data, writer)

        lr_schedulers.step(epoch)

        if epoch % 2 == 0:
            def_loss, total_loss, ter_accs, pro_accs = run_epoch(npi, 'eval', eval_data, writer)
            if def_loss < Best_results['def_loss']:
                Best_results['def_loss'] = def_loss
                Best_results['epoch_def_loss'] = epoch
            if total_loss < Best_results['total_loss']:
                Best_results['total_loss'] = total_loss
                Best_results['epoch_total_loss'] = epoch
            if ter_accs > Best_results['ter_accs']:
                Best_results['ter_accs'] = ter_accs
                Best_results['epoch_ter_accs'] = epoch
            if pro_accs > Best_results['pro_accs']:
                Best_results['pro_accs'] = pro_accs
                Best_results['epoch_pro_accs'] = epoch

            torch.save(npi.state_dict(), os.path.join(exp_dir, 'npi_{}.pth'.format(epoch)))

    for key, val in Best_results.items():
        if key.find('epoch') != -1:
            print('Best %s for test same at %d epoch' % (key, val))
        else:
            print('Best %s for test same: %f' % (key, val))

