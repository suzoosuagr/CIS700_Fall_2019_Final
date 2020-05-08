import math
import torch
import torch.nn as nn
from Env.maze import CONFIG

class MazeCore(nn.Module):
    def __init__(self, hidden_dim=256, state_dim=128):
        super(MazeCore, self).__init__()
        self.hidden_dim, self.state_dim = hidden_dim, state_dim
        # 4*3 + 2*100 + 2*11, # 5*5*3 + 2*25 + 2*6 = 137
        self.env_dim = CONFIG["ENV_DIM"]
        # 2 * 11 = 22, 1 * 12 = 12
        self.arg_dim = CONFIG["ARG_NUM"] * CONFIG["ARG_DEPTH"]
        self.pro_dim = CONFIG["PRO_EMBED_DIM"]

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.env_dim + self.arg_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.state_dim),
        )

        tmp = torch.empty((CONFIG["PRO_NUM"], CONFIG["PRO_KEY_DIM"]), dtype=torch.float32).normal_(mean=0, std=1)
        self.pro_key = nn.Parameter(tmp, requires_grad=True)

        self.pro_embedder = nn.Embedding(CONFIG["PRO_NUM"], self.pro_dim)

    def forward(self, env_in, arg_in, prg_in):
        state_in = torch.cat((env_in, arg_in), 1).float()

        state_ft = self.encoder(state_in)

        prg_in = prg_in.long()
        pro_id_embed = self.pro_embedder(prg_in)

        return state_ft, pro_id_embed

# x = MazeCore()
