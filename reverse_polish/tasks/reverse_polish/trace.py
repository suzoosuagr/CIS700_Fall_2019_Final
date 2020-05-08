from tasks.reverse_polish.config import ScratchPad, I2A, PROGRAM_ID as P
MOV_PTR, WRITE, REVPOLI, PRECE, PUSH, POP = "MOV_PTR", "WRITE", "REVPOLI", "PRECE", "PUSH", "POP"
WRITE_STACK = 0 
WRITE_OUT = 1
EXP_PTR, STACK_PTR, OUT_PTR = range(3)
LEFT, RIGHT = 0, 1

class Trace():
    def __init__(self, exp, debug=False):
        super().__init__()
        self.exp = exp
        """
        generate and printting trace
        """ 
        self.trace = []
        self.scratch = ScratchPad(exp, debug=debug)
        self.build()

    def build(self):
        self.trace.append(((REVPOLI, P[REVPOLI]), [], False))

        while not self.scratch.done():
            self.precedence()
            self.next()
        
    def precedence(self):
        self.trace.append(((PRECE, P[PRECE]), [], False))
        push_flg, value = self.scratch.prece()
        if push_flg == 0:
            self.push(value)
        elif push_flg == 1:
            self.pop()
            self.push(value)
        elif push_flg == 2:
            self.write_out(value)
        elif push_flg == 3:
            self.pop_loop()

    def write_out(self, value):
        self.trace.append(((WRITE, P[WRITE]), [WRITE_OUT, value], False))
        self.trace.append(((MOV_PTR, P[MOV_PTR]), [OUT_PTR, RIGHT], False))
        
        self.scratch.write_out(value)

    def push(self, value):
        self.trace.append(((MOV_PTR, P[MOV_PTR]), [STACK_PTR, RIGHT], False))
        self.trace.append(((WRITE, P[WRITE]), [STACK_PTR, value], False))

        self.scratch.push(value)

    def pop(self):
        # read 
        s_top = self.scratch.pop_read()
        if I2A[s_top] not in [*'()']:
            self.write_out(s_top)
        # write
        self.trace.append(((WRITE, P[WRITE]), [STACK_PTR, 0], False))
        self.trace.append(((MOV_PTR, P[MOV_PTR]), [STACK_PTR, LEFT], False))
        
        self.scratch.pop()

    def pop_loop(self):
        s_top = self.scratch.pop_read()
        if I2A[s_top] not in [*'()']:
            self.write_out(s_top)

        self.trace.append(((WRITE, P[WRITE]), [STACK_PTR, 0], False))
        self.trace.append(((MOV_PTR, P[MOV_PTR]), [STACK_PTR, LEFT], False))

        self.scratch.pop()

        if I2A[s_top] != '(':
            self.pop_loop()

    def pop_all(self):
        s_top = self.scratch.pop_read()
        if s_top == 0:
            self.trace.append(((MOV_PTR, P[MOV_PTR]), [STACK_PTR, LEFT], True))
            return 
        if I2A[s_top] not in [*'()']:
            self.write_out(s_top)
        
        self.trace.append(((WRITE, P[WRITE]), [STACK_PTR, 0], False))
        self.trace.append(((MOV_PTR, P[MOV_PTR]), [STACK_PTR, LEFT], False))

        self.scratch.pop()

        self.pop_all()

    def next(self):
        self.trace.append(((MOV_PTR, P[MOV_PTR]), [EXP_PTR, RIGHT], False))
        self.scratch.next()
        if self.scratch.done():
            # self.trace.append(((MOV_PTR, P[MOV_PTR]), [OUT_PTR, RIGHT], True))
            self.pop_all()
        else:
            pass




