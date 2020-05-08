import pickle
import numpy as np
import random
import os
from collections import deque
from Env.maze import Maze_Env, PROGRAM_ID

Push = 'Push'
Pop = 'Pop'
Empty = 'Empty'
Visit = 'Visit'
Arrive = 'Arrive'
Fail = 'Fail'
Success = 'Success'
Check = 'Check'

def solve_maze(maze, start, end):
    trace = []
    num_rows = maze.shape[0]
    num_cols = maze.shape[1]
    visited = np.zeros((num_rows, num_cols), dtype=np.bool)

    dir_values = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    queue = deque()
    queue.append(start)
    trace.append(((Push, PROGRAM_ID[Push]), [start], False)) # Push

    trace.append(((Empty, PROGRAM_ID[Empty]), [], False))  # if Empty
    while queue:
        cur = queue.popleft()
        trace.append(((Pop, PROGRAM_ID[Pop]), [], False)) # Pop

        visited[cur[0], cur[1]] = True
        trace.append(((Visit, PROGRAM_ID[Visit]), [cur], False)) # Visit

        trace.append(((Arrive, PROGRAM_ID[Arrive]), [cur], False)) # if arrive
        if cur[0] == end[0] and cur[1] == end[1]:
            trace.append(((Success, PROGRAM_ID[Success]), [], True))  # success, update result
            return trace

        for val in dir_values:
            new_r = cur[0] + val[0]
            new_c = cur[1] + val[1]
            new = (new_r, new_c)
            trace.append(((Check, PROGRAM_ID[Check]), [new], False))

            if 0 <= new_r < num_rows and 0 <= new_c < num_cols:
                # print(maze[new_r, new_c] )
                # print(visited[new_r, new_c])
                if maze[new_r, new_c] == 0 and visited[new_r, new_c] == False:
                    queue.append(new)
                    trace.append(((Push, PROGRAM_ID[Push]), [new], False))

    trace.append(((Fail, PROGRAM_ID[Fail]), [], True))  # fail, update result
    return trace

def generate_maze(num_rows, num_cols):
    # 0 is blank and unvisited, 1 is wall, 2 is visited.
    maze = np.ones((num_rows, num_cols), dtype=np.uint8)
    # build walls for each cell.
    for i in range(0, num_rows, 2):
        for j in range(0, num_cols, 2):
            maze[i, j] = 0
    # DFS from (0, 0)
    r = 0
    c = 0
    # visited cell
    history = [(r, c)]

    # up, down, left, right
    dir_values = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    paths = []
    while history:
        if maze[r, c] != 2:
            maze[r, c] = 2
            paths.append((r, c))

        candidates = []
        for val in dir_values:
            new_r = r + val[0]
            new_c = c + val[1]
            if 0 <= new_r < num_rows and 0 <= new_c < num_cols:
                if maze[new_r, new_c] == 0:
                    candidates.append(val)

        if len(candidates):
            history.append((r, c))
            val = random.choice(candidates)
            new_r = r + val[0]
            new_c = c + val[1]
            # set wall to path and connect cells
            maze[(r+new_r)//2, (c+new_c)//2] = 2
            paths.append(((r+new_r)//2, (c+new_c)//2))
            r = new_r
            c = new_c
        else:
            r, c = history.pop()
    slice = random.sample(paths, 2)
    start = slice[0]
    end = slice[1]
    for i in range(num_rows):
        for j in range(num_cols):
            if maze[i,j] == 2:
                maze[i, j] = 0

    return maze, start, end

def generate_data(prefix, num, size):
    data = []
    for i in range(num):
        maze, start, end = generate_maze(size, size)
        trace = solve_maze(maze, start, end)
        data.append((start, end, maze, trace))

    print(os.getcwd())
    with open('%s.pik' % prefix, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    generate_data('train_5_50', 50, 5)
    generate_data('train_5_100', 100, 5)
    generate_data('train_7_100', 100, 7)
    generate_data('test_5_50', 50, 5)
    generate_data('test_7_50', 50, 7)
    generate_data('test_10_50', 50, 10)

    # DATA_PATH = 'train.pik'
    # with open(DATA_PATH, 'rb', ) as f:
    #     data = pickle.load(f)
    # z = 1