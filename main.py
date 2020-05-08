import numpy as np
from tasks.reverse_polish.generate_data import generate_rev_polish

ARGS = {
    "GENERATE": True, 
    "DEBUG": True,
    "MODE": 'train', 
}

if __name__ == "__main__":
    if ARGS["GENERATE"]:
        generate_rev_polish('train', 2000, ARGS["DEBUG"])
        generate_rev_polish('eval', 200, ARGS["DEBUG"])
        generate_rev_polish('test', 500, ARGS["DEBUG"])

    # eval(ARGS["MODE"])()

    print("FINISHED!")


