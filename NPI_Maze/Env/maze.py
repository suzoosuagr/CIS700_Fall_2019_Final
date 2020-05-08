import numpy as np
from collections import deque


CONFIG = {
    # 4*3 (4 neighbor of unvis, wall, vis)
    # + 10*10(head) + 10*10(tail)
    # + 2*11(end x,y)
    "ENV_DIM": 4*3 + 2*100 + 2*11,

    "ARG_NUM": 1,            # Maximum Number of Program Arguments, 1 node
    "ARG_DEPTH": 22,         # Size of Argument Vector => One-Hot, Options 0-10, 9 + 1(invaild), x and y
    "DEF_ARG_VALUE": 10,      # Default Argument Value

    "PRO_NUM": 8,             # Maximum Number of Subroutines
    "PRO_KEY_DIM": 5,        # Size of the Program Keys
    "PRO_EMBED_DIM": 10  # Size of the Program Embeddings
}

PROGRAM_SET = [
    ('Push', 5, 5), #0
    ('Pop',), #1
    ('Empty',), #2
    ('Visit', 5, 5), #3
    ('Arrive', 5, 5), #4
    ('Fail',), #5
    ('Success',), #6
    ('Check', 5, 5), #7
]

PROGRAM_ID = {x[0]: i for i, x in enumerate(PROGRAM_SET)}

class Maze_Env():
    def __init__(self, start, end, maze):
        self.maze = maze
        self.num_rows = maze.shape[0]
        self.num_cols = maze.shape[1]
        self.visited = np.zeros((self.num_rows, self.num_cols), dtype=np.uint8)
        self.queue = deque()

        self.start = start
        self.end = end
        self.cur = start
        self.head = self.tail = 0 # Maximum = 24 (0-24)
        self.result = False

    def execute(self, pro_id, args):
        if pro_id == 0: # Push
            cur = args[0]
            self.queue.append(cur)
            self.tail += 1
        elif pro_id == 1: # Pop
            self.cur = self.queue.popleft()
            self.head += 1
        elif pro_id == 2: # Empty
            pass
        elif pro_id == 3: # Visit
            cur = args[0]
            self.visited[cur[0], cur[1]] = 1
        elif pro_id == 4: # Arrive
            pass
        elif pro_id == 5: # Fail
            self.result = False
        elif pro_id == 6: # Success
            self.result = True
        elif pro_id == 7: # Check
            pass

    def encode_env(self):
        # env_ft_1 = np.zeros((self.num_rows, self.num_cols, 3), dtype=np.int32)
        # for i in range(self.num_rows):
        #     for j in range(self.num_cols):
        #         if self.maze[i,j] == 1: # wall
        #             env_ft_1[i, j, 1] = 1
        #         elif self.visited[i,j] == 1: # visited
        #             env_ft_1[i, j, 2] = 1
        #         else: # unvisited
        #             env_ft_1[i, j, 0] = 1
        # env_ft_1 = env_ft_1.flatten()

        #(4 neighbor of unvis, wall, vis)
        env_ft_1 = np.zeros((4, 3), dtype=np.int32)
        dir_values = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for idx, val in enumerate(dir_values):
            new_r = self.cur[0] + val[0]
            new_c = self.cur[1] + val[1]

            if 0 <= new_r < self.num_rows and 0 <= new_c < self.num_cols:
                if self.maze[new_r, new_c] == 0 and self.visited[new_r, new_c] == 0:
                    env_ft_1[idx, 0] = 1
                else:
                    env_ft_1[idx, 2] = 1
            else:
                env_ft_1[idx, 1] = 1
        env_ft_1 = env_ft_1.flatten()

        env_ft_2 = np.zeros((2, 10 * 10), dtype=np.int32)
        env_ft_2[0, self.head] = 1
        env_ft_2[1, self.tail] = 1
        env_ft_2 = env_ft_2.flatten()

        env_ft_3 = np.zeros((2, CONFIG['ARG_DEPTH'] // 2), dtype=np.int32)
        env_ft_3[0, self.end[0]] = 1
        env_ft_3[1, self.end[1]] = 1
        env_ft_3 = env_ft_3.flatten()

        env_ft = np.concatenate((env_ft_1, env_ft_2, env_ft_3))

        return env_ft

    def encode_args(self, args):
        arg_ft = np.zeros((2, CONFIG['ARG_DEPTH'] // 2), dtype=np.int32)
        if len(args) > 0:
            cur = args[0]
            if 0 <= cur[0] < self.num_rows and 0 <= cur[1] < self.num_cols:
                # if self.maze[cur[0], cur[1]] == 0 and self.visited[cur[0], cur[1]] == 0:
                arg_ft[0, cur[0]] = 1
                arg_ft[1, cur[1]] = 1
            else:
                arg_ft[0, -1] = 1
                arg_ft[1, -1] = 1
        else:
            arg_ft[0, -1] = 1
            arg_ft[1, -1] = 1
        arg_ft = arg_ft.flatten()
        return arg_ft