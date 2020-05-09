import math
import torch
import torch.nn as nn
import os

class NPI(nn.Module):
    def __init__(self, core, config, npi_core_dim=256, npi_core_layer=2):
        super(NPI, self).__init__()

        self.core = core
        self.config = config
        self.npi_core_dim = npi_core_dim
        self.npi_core_layer = npi_core_layer

        self.num_args = config["ARGUMENT_NUM"]
        self.arg_dim = config["ARGUMENT_DEPTH"]
        self.num_prog  = config["PROGRAM_NUM"]
        self.key_dim   = config["PROGRAM_KEY_SIZE"]

        self.terminate_decoder = nn.Linear(npi_core_dim, 2)
        self.arguments_decoder = Arg_Net(self.npi_core_dim, self.arg_dim)
        self.programes_decoder = Key_Net(self.npi_core_dim, self.key_dim, self.num_prog)

        self.npi_core = NPI_Core(self.core.state_dim, self.core.program_dim, self.npi_core_dim, self.npi_core_layer)


    def forward(self, env_in, arg_in, prg_in, h0, initial=False):

        state_encoding, program_embedding = self.core(env_in, arg_in, prg_in)

        if initial:

            out, hidden = self.npi_core(state_encoding, program_embedding, h0, initial)
        else:
            out, hidden = self.npi_core(state_encoding, program_embedding, h0, initial)

        ter = self.terminate_decoder(out)
        prog = self.programes_decoder(self.core.program_key, out)
        args = self.arguments_decoder(out)

        return (ter, prog, args), hidden

class NPI_Core(nn.Module):
    def __init__(self, state_dim, program_dim, npi_core_dim=256, npi_core_layer=2):
        super(NPI_Core, self).__init__()

        self.rnn = nn.GRU(state_dim + program_dim, npi_core_dim, npi_core_layer)

    def forward(self, state_encoding, program_embedding, h0, initial):
        state_encoding = state_encoding.unsqueeze(0)           # [len, b, dim]
        c = torch.cat([state_encoding, program_embedding], dim=2)
        if initial:
            out, hn = self.rnn(c, h0)
        else:
            out, hn = self.rnn(c)
        return out, hn

class Key_Net(nn.Module):
    def __init__(self, npi_core_dim=256, key_dim=5, num_program=6):
        super(Key_Net, self).__init__()

        self.num_program = num_program

        self.fc = nn.Sequential(
            nn.Linear(npi_core_dim, key_dim),
            nn.ReLU(),
            nn.Linear(key_dim, key_dim)
        )
    
    def forward(self, program_key, hidden):
        # program_key with shape : [program_num, PROGRAM_KEY_SIZE]
        program_key = program_key.unsqueeze(0)
        hidden = hidden.squeeze(0)
        key = self.fc(hidden)
        key = key.repeat(1, self.num_program ,1) # [1, num_program, PROGRAM_KEY_SIZE]

        prog_sim = key * program_key  # [1, num_program, PROGRAM_KEY_SIZE]
        prog_dist = torch.sum(prog_sim, dim=2)
        return prog_dist

class Arg_Net(nn.Module):
    def __init__(self, npi_core_dim=256, arg_dim=15):
        super(Arg_Net, self).__init__()

        self.arg_fc0 = nn.Sequential(
            nn.Linear(npi_core_dim, arg_dim),
            nn.ReLU()
        )

        self.arg_fc1 = nn.Sequential(
            nn.Linear(npi_core_dim, arg_dim),
            nn.ReLU()
        )

    def forward(self, hidden):
        arg0 = self.arg_fc0(hidden)
        arg1 = self.arg_fc1(hidden)

        return (arg0, arg1)


class NPI_LOSS(nn.Module):
    def __init__(self, alpha=2, beta=0.25, batch_size=1):
        super(NPI_LOSS, self).__init__()

        self.batch = batch_size
        self.criterion = nn.CrossEntropyLoss()

        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        ter, prog, args = pred
        ter_y, prog_y, args_y = gt
        arg0, arg1 = args
        args_y = args_y.view(2, -1)
        arg0_y = torch.argmax(args_y[0]).long()
        arg1_y = torch.argmax(args_y[1]).long()


        ter_loss = self.criterion(ter.squeeze(0), ter_y)
        prog_loss = self.criterion(prog, prog_y)
        arg0_loss = self.criterion(arg0.squeeze(0), arg0_y.view(1))
        arg1_loss = self.criterion(arg1.squeeze(0), arg1_y.view(1))

        default_loss = self.alpha * ter_loss + prog_loss
        arg_loss     = self.beta  * (ter_loss + prog_loss) + (arg0_loss + arg1_loss)

        return default_loss, arg_loss

    def metric(self, pred, gt):
        ter, prog, args = pred
        ter_y, prog_y, args_y = gt
        arg0, arg1 = args
        args_y = args_y.view(2, -1)
        arg0_y = torch.argmax(args_y[0]).long()
        arg1_y = torch.argmax(args_y[1]).long()

        _, pro_pred_idx = torch.max(prog, 1)
        pro_re = (pro_pred_idx == prog_y).squeeze()
        pro_acc = 0.0 + pro_re.item()

        _, ter_pred_idx = torch.max(ter.squeeze(0), 1)
        ter_re = (ter_pred_idx == ter_y).squeeze()
        ter_acc = 0.0 + ter_re.item()

        return pro_acc, ter_acc




