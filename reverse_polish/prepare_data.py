import numpy as np
from tasks.reverse_polish.generate_data import generate_rev_polish

ARGS = {
    "GENERATE": True, 
    "DEBUG": True,
}

if __name__ == "__main__":
    if ARGS["GENERATE"]:
        generate_rev_polish('train', 500, ARGS["DEBUG"])
        generate_rev_polish('eval', 50, ARGS["DEBUG"])
        generate_rev_polish('test', 50, ARGS["DEBUG"])
        print("FINISHED Generating")


    

    