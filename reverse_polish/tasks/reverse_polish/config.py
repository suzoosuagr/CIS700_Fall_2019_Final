import numpy as np
import sys
import time

CONFIG = {
    "ENV_ROW": 3,
    "ENV_COL": 15,
    "ENV_DEP": 15,

    "ARGUMENT_NUM": 2,
    "ARGUMENT_DEPTH": 15, 
    "DEFAULT_ARGUMENT_VALUE": 0,

    "PROGRAM_NUM": 6,
    "PROGRAM_KEY_SIZE": 5, 
    "PROGRAM_EMBEDDING_SIZE":10

}

PROGRAM_SET = [
    ("MOV_PTR", 3, 2),   # 3 options with 2 mode left and right
    ("WRITE", 2, 15),       # Write charactor to out, 15 options, 0 for stack, 1 for out
    ("REVPOLI", ),       # TOP-level reverse polish program
    ("PRECE", ),        # compare the precedence
    ("PUSH", 15),       # push value to stack 
    ("POP",)
]
PROGRAM_ID = {x[0]:i for i, x in enumerate(PROGRAM_SET)}

Alpha = [*'.ABCDEFGH()+-*/']
Index = range(0, len(Alpha)+1)
A2I = dict(zip(Alpha, Index))
I2A = dict(zip(Index, Alpha))
PRECEDENCE = dict(zip([*'*/+-'], [3, 3, 2, 2]))

class ScratchPad():
    def __init__(self, exp, rows=CONFIG['ENV_ROW'], cols=CONFIG['ENV_COL'], debug=False):
        self.rows, self.cols = rows, cols
        self.scratchpad = np.zeros((self.rows, self.cols), dtype=np.int8)

        self.init_scratchpad(exp)
        self.exp_ptr, self.stack_ptr, self.out_ptr = self.ptrs = [(x, 0) for x in range(self.rows)]
        self.stack_ptr = (self.stack_ptr[0], self.stack_ptr[1] - 1)
        self.len_exp = len(exp)

        self.debug = debug

    def init_scratchpad(self, exp):
        """
        Initial the scratchpad for reverse polish
        """
        for i in range(len(exp)):
            self.scratchpad[0, i] = A2I[exp[i]]

    def done(self):
        if self.exp_ptr[1] > (self.cols - 1):
            return True
        else:
            if self.exp_ptr[1] > (self.len_exp - 1):
                return True
            else:
                return False

    def prece(self):
        """
        check the precedence of operations. 
        """
        exp_pre = 0
        stack_pre = 0
        if self[self.exp_ptr] == A2I['(']:
            exp_pre = 4
        if self[self.stack_ptr] == A2I['(']:
            stack_pre = 1

        if self[self.exp_ptr] == A2I[')']:
            return 3, self[self.exp_ptr]
        
        if exp_pre == 0:
            try:
                exp_pre = PRECEDENCE[I2A[self[self.exp_ptr]]]
            except KeyError:
                exp_pre = -1  # operand direct to out. 
        if stack_pre == 0:
            try:
                stack_pre = PRECEDENCE[I2A[self[self.stack_ptr]]]
            except KeyError:
                pass

        if stack_pre < exp_pre:
            code, value =  0, self[self.exp_ptr]    # push
        elif exp_pre != -1:
            code, value =  1, self[self.exp_ptr]     # pop
        else:
            code, value = 2, self[self.exp_ptr]     # direct write exp to out

        return code, value


    def write_stack(self, value):
        self[self.stack_ptr] = value
        if self.debug:
            self.debugger()
    
    def write_out(self, value):
        self[self.out_ptr] = value
        self.out_ptr = (self.out_ptr[0], self.out_ptr[1] + 1)
        if self.debug:
            self.debugger()
        
    def push(self, value):
        self.stack_ptr = (self.stack_ptr[0], self.stack_ptr[1] + 1)
        self.write_stack(value)
        # self.stack_ptr = (self.stack_ptr[0], self.stack_ptr[1] + 1)
        
    def pop(self):
        self.write_stack(0)
        self.stack_ptr = (self.stack_ptr[0], self.stack_ptr[1] - 1)
    
    def pop_read(self):
        s_top = self[self.stack_ptr]
        return s_top

    def next(self):
        self.exp_ptr = (self.exp_ptr[0], self.exp_ptr[1] + 1)
        
    def get_env(self):
        """
        get the value of registers. 
        """
        env = np.zeros((CONFIG["ENV_ROW"], CONFIG["ENV_DEP"]), dtype=np.int32)
        if self.exp_ptr[1] > self.cols - 1:
            env[0][0] = 1
        else:
            env[0][self[self.exp_ptr]] = 1
        if self.stack_ptr[1] > self.cols - 1:
            env[1][0] = 1
        else:
            env[1][self[self.stack_ptr]] = 1
        if self.out_ptr[1] > self.cols - 1:
            env[2][0] = 1
        else:
            env[2][self[self.out_ptr]] = 1
        return env.flatten()

    def debugger(self):
        i2a = lambda x: [I2A[i] for i in x]
        new_strs = ["".join(i2a(self[i])) for i in range(3)]
        line_length = len('Expression:' + " " * 5 + new_strs[0])
        print('Expression:' + " " * 5 + new_strs[0])
        print('Stack     :' + " " * 5 + new_strs[1])
        print('-' * line_length)
        print('Rev Polish:' + " " * 5 + new_strs[2])
        print('#' * line_length)
        time.sleep(.1)
        sys.stdout.flush()
        
    def execute(self, prog_id, args):
        if prog_id == 0:   # MOVE
            ptr, lr = args
            lr = (lr * 2) - 1
            if ptr == 0:   # exp_ptr
                self.exp_ptr = (self.exp_ptr[0], self.exp_ptr[1] + lr)
            elif ptr == 1: # stack_ptr
                self.stack_ptr = (self.stack_ptr[0], self.stack_ptr[1] + lr)
            elif ptr == 2:  # out_ptr
                self.out_ptr = (self.out_ptr[0], self.out_ptr[1] + lr)
            else:
                raise NotImplementedError
            self.ptrs = [self.exp_ptr, self.stack_ptr, self.out_ptr]
        elif prog_id == 1: # Write
            ptr, val = args
            if ptr == 0:   # Write Stack
                self[self.stack_ptr] = val
            if ptr == 1:   # Write Out
                self[self.out_ptr] = val

    def encode_args(self, args):
        arg_vec = np.zeros((CONFIG["ARGUMENT_NUM"], CONFIG['ARGUMENT_DEPTH']), dtype=np.int32)
        if len(args) > 0:
            for i in range(CONFIG["ARGUMENT_NUM"]):
                if i >= len(args):
                    arg_vec[i][CONFIG["DEFAULT_ARGUMENT_VALUE"]] = 1
                else:
                    arg_vec[i][args[i]] = 1
        else:
            for i in range(CONFIG["ARGUMENT_NUM"]):
                arg_vec[i][CONFIG["DEFAULT_ARGUMENT_VALUE"]] = 1
        return arg_vec.flatten()

    def __getitem__(self, item):
        return self.scratchpad[item]

    def __setitem__(self, key, value):
        self.scratchpad[key] = value
