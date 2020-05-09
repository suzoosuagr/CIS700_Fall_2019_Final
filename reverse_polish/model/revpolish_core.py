import math
import torch
import torch.nn as nn
from tasks.reverse_polish.config import CONFIG

class RevPolishCore(nn.Module):
    def __init__(self, hidden_dim=256, state_dim=128):
        super(RevPolishCore, self).__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.env_dim = CONFIG["ENV_ROW"] * CONFIG["ENV_DEP"]
        self.arg_dim = CONFIG["ARGUMENT_NUM"] * CONFIG["ARGUMENT_DEPTH"]
        self.program_dim = CONFIG["PROGRAM_EMBEDDING_SIZE"]

        # Eev encoder. 
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.env_dim + self.arg_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.state_dim)
        )

        # tmp = torch.empty((CONFIG["PROGRAM_NUM"], CONFIG["PROGRAM_KEY_SIZE"]))
        self.program_key = torch.randn((CONFIG["PROGRAM_NUM"], CONFIG["PROGRAM_KEY_SIZE"]), requires_grad=True).float().cuda()

        # Program embedder
        self.program_embd = nn.Embedding(CONFIG["PROGRAM_NUM"], CONFIG["PROGRAM_EMBEDDING_SIZE"])

    def forward(self, env_in, arg_in, prg_in):
        state_in = torch.cat([env_in, arg_in], 1).float()
        state_en = self.encoder(state_in)

        prg_in = prg_in.long()
        prg_id_embd = self.program_embd(prg_in)
        return state_en, prg_id_embd

