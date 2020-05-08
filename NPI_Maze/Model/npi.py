import math
import torch
import torch.nn as nn
import os

class NPI(nn.Module):
    def __init__(self, core, config, npi_core_dim=256, npi_core_layers=2):
        super(NPI, self).__init__()

        self.core = core
        self.state_dim = core.state_dim
        self.pro_dim = core.pro_dim
        self.npi_core_dim = npi_core_dim
        self.npi_core_layers = npi_core_layers
        self.num_args, self.arg_depth = config["ARG_NUM"], config["ARG_DEPTH"]
        self.num_progs, self.key_dim = config["PRO_NUM"], config["PRO_KEY_DIM"]

        # Build NPI LSTM Core, hidden state
        self.lstm = self.npi_core()

        # Build Termination Network => Returns probability of terminating
        self.ter_net = self.build_ter_net()

        # Build Key Network => Generates probability distribution over programs
        self.pro_net = self.build_pro_net()

        # Build Argument Networks => Generates list of argument distributions
        for i in range(self.num_args):
            setattr(self, 'arg_net_%d' % i, self.build_arg_net())

        # self.arg_net = self.build_arg_net()

    def init_state(self, batch_size):
        """
        Zero NPI Core LSTM Hidden States. LSTM States are represented as a Tuple, consisting of the
        LSTM C State, and the LSTM H State (in that order: (c, h)).
        """
        hidden_state = torch.zeros(batch_size, 2 * self.npi_core_dim)
        return hidden_state

    def npi_core(self):
        network = torch.nn.LSTM(self.state_dim + self.pro_dim,
                                self.npi_core_dim,
                                self.npi_core_layers)
        return network

    def build_ter_net(self):
        network = torch.nn.Sequential(
            torch.nn.Linear(self.npi_core_dim, self.npi_core_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.npi_core_dim // 2, 2),
        )

        return network

    def build_pro_net(self):
        network = torch.nn.Sequential(
            torch.nn.Linear(self.npi_core_dim, self.key_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.key_dim, self.key_dim),
        )

        return network

    def build_arg_net(self):
        network = torch.nn.Sequential(
            torch.nn.Linear(self.npi_core_dim, self.arg_depth),
            torch.nn.ReLU(),
            torch.nn.Linear(self.arg_depth, self.arg_depth),
        )
        return network

    def forward(self, env_in, arg_in, prg_in):
        b, _, = env_in.size()
        static_ft, pro_id_embed = self.core(env_in, arg_in, prg_in)
        static_ft = torch.unsqueeze(static_ft, 1)
        merge_ft = torch.cat((static_ft, pro_id_embed), dim=2)
        merge_ft = merge_ft.permute(1, 0, 2)
        # hidden = self.init_state(b)

        lstm_out, (hn, cn) = self.lstm(merge_ft)
        lstm_out = lstm_out[-1]
        ter_out = self.ter_net(lstm_out)
        key_out = self.pro_net(lstm_out).view((-1, 1, self.key_dim))
        key_out = key_out.repeat(1, self.num_progs, 1)
        z = self.core.pro_key
        pro_sim = key_out * z
        # pro_sim = torch.matmul(key_out, self.core.pro_key)
        pro_dist = torch.sum(pro_sim, dim=2)

        args = []
        for i in range(self.num_args):
            cur_arg_net = getattr(self, 'arg_net_%d' % i)
            arg_out = cur_arg_net(lstm_out)
            args.append(arg_out)

        args = torch.cat(args, 1)

        return pro_dist, args, ter_out

    def cal_loss(self, pred, gt):
        pro_pred, arg_pred, ter_pred = pred
        pro_out_ft, arg_out_ft, ter_out_ft = gt
        ter_out_ft = ter_out_ft.long()
        pro_out_ft = pro_out_ft.long()

        ce = nn.CrossEntropyLoss()
        ter_loss = ce(ter_pred, ter_out_ft)
        pro_loss = ce(pro_pred, pro_out_ft)

        l1 = nn.L1Loss()
        arg_out_ft = arg_out_ft.float()
        arg_loss = l1(arg_pred, arg_out_ft)

        default_loss = 2 * ter_loss + pro_loss
        total_loss = 0.25 * (ter_loss + pro_loss) + arg_loss

        return default_loss, total_loss

    def cal_metrics(self, pred, gt):
        pro_pred, arg_pred, ter_pred = pred
        pro_out_ft, arg_out_ft, ter_out_ft = gt


        _, pro_pred_idx = torch.max(pro_pred, 1)
        pro_re = (pro_pred_idx == pro_out_ft).squeeze()
        pro_acc = 0.0 + pro_re.item()

        _, ter_pred_idx = torch.max(ter_pred, 1)
        ter_re = (ter_pred_idx == ter_out_ft).squeeze()
        ter_acc = 0.0 + ter_re.item()

        return pro_acc, ter_acc

    def save_network(self, save_path, cuda_flag):
        if cuda_flag:
            torch.save(self.cpu().state_dict(), save_path)
        else:
            torch.save(self.state_dict(), save_path)

    def load_networks(self, load_path):
        self.load_state_dict(torch.load(load_path))
