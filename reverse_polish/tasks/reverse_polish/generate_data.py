import numpy as np
from tasks.reverse_polish.trace import Trace
import pickle
import random 

# FIX SEED
random.seed(1)
np.random.seed(1)

MAX_LENGTH = 8
OPERAND = [*'ABCDEFGH']
OPERATOR = [*'+-*/']
SPECIAL = [*'()']

def generate_rev_polish(name, num_examples, debug=False, debug_freq = 1000):
    data = []
    for i in range(num_examples):
        exp = get_expressions()
        while len(exp) > MAX_LENGTH:
            exp = get_expressions()
        if debug and i % debug_freq == 0:
            trace = Trace(exp, debug=True).trace
        else:
            trace = Trace(exp, debug=False).trace
        data.append((exp, trace))

    with open('tasks/reverse_polish/data/{}.pik'.format(name), 'wb') as f:
        pickle.dump(data, f)



def get_expressions():
    num_par = np.random.choice(range(1, 4), 1)

    exp_dict = {0:[], 1:[], 2:[]}
    for i in range(num_par[0]):
        num_operands = np.random.choice([2, 3, 4], 1)
        num_operators = num_operands - 1

        operands = np.random.choice(OPERAND, num_operands, replace=False)
        operators = np.random.choice(OPERATOR, num_operators, replace=True)

        sub_exp = ['.' for _ in range(num_operators[0] + num_operands[0])]
        for j in range(num_operators[0] + num_operands[0]):
            if j % 2 == 0:
                sub_exp[j] = operands[j//2]
            if j % 2 != 0:
                sub_exp[j] = operators[(j-1)//2]
        exp_dict[i] = ''.join(sub_exp)

    flg = np.random.choice(range(2), 1)
    if flg[0] == 0:                     # parallel
        if num_par[0] == 1:
            final_exp = exp_dict[0]
        elif num_par[0] == 2:
            operators = np.random.choice(OPERATOR, 1)
            final_exp = operators[0].join([exp_dict[0], exp_dict[1]])
        else:
            operators = np.random.choice(OPERATOR, 2)
            final_exp = operators[0].join([exp_dict[0], exp_dict[1]])
            final_exp = operators[1].join([final_exp, exp_dict[2]])
            
    else:
        if num_par[0] == 1:             # involving 
            final_exp = exp_dict[0]
        elif num_par[0] == 2:
            operators = np.random.choice(OPERATOR, 1)
            subexp = '({})'.format(exp_dict[1])
            final_exp = operators[0].join([exp_dict[0], subexp])
        else:
            operators = np.random.choice(OPERATOR, 2)
            subexp = '({})'.format(exp_dict[1])
            subexp = operators[0].join(['', subexp])
            subsub = '({})'.format(''.join([exp_dict[2], subexp]))
            subsub = operators[1].join(['', subsub])
            final_exp = ''.join([exp_dict[0], subsub])

    return final_exp