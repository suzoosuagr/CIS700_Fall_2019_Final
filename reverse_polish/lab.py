from tasks.reverse_polish.generate_data import get_expressions
from tasks.reverse_polish.trace import Trace

for i in range(5):
    exp = get_expressions()
    while len(exp) > 14:
        exp = get_expressions()

    print("TRACE")
    trace = Trace(exp, debug=True).trace
    print("_"*10)
    print(trace)
