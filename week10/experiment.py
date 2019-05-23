import os
import json

ARGUMENT_FILE = 'argument.json'

with open (ARGUMENT_FILE) as a:
    exp_count = 1
    arg = json.load(a)
    arguments = arg["arguments"]
    for exp in arguments:
        cmd = "python3 "
        os.system(cmd+)
        exp_count = exp_count + 1
