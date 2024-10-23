import json

def loadConfig(file):
    f = open(file)
    j = json.load(f)
    f.close
    return j
