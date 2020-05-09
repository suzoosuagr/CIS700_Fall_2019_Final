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
        self.arguments_decoder = Arg_Net(self.npi_core_layer, self.arg_dim)
        self.programes_decoder = Key_Net(self.npi_core_layer, self.key_dim, self.num_prog)

        self.npi_core = NPI_Core(self.core.state_dim, self.core.program_dim, self.npi_core_dim, self.npi_core_layer)


    def forward(self, ):




class NPI_Core(nn.Module):
    def __init__(self, state_dim, program_dim, npi_core_dim=256, npi_core_layer=2):
        super(NPI_Core, self).__init__()

        self.rnn = nn.GRU(state_dim + program_dim, npi_core_dim, npi_core_layer)

    def forward(self, state_encoding, program_embedding, h0):
        c = torch.cat([state_encoding, program_embedding], dim=2)
        out, hn = rnn(input, h0)
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