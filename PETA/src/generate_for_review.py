# Genera un montón de imágenes grad cam para poder comparar y decidir

from config import loadConfig
from test import testExperiment
from util import getClassDescription

num_classes = 3
config = loadConfig("config.muestraFull3700_4_test.json")
weights = "/Users/hugom/Tesis/ExperimentosServer/muestraFull3700_4/MuestraFull3700_4_server_0_epoch27.pth"
#studyID = "I30507" #AD
#studyID = "I153776" # CN
studyID = "I105273" # MCI

# TODO: elegir algunos para 2 clases

layerMatrix = [
    ["Conv2d_1a_3x3"],
    ["Conv2d_2a_3x3"],
    ["Conv2d_2b_3x3"],
    ["maxpool1"],
    ["Conv2d_3b_1x1"],
    ["Conv2d_4a_3x3"],
    ["maxpool2"],
    ["Mixed_5b"],
    ["Mixed_5c"],
    ["Mixed_5d"],
    ["Mixed_6a"],
    ["Mixed_6b"],
    ["Mixed_6c"],
    ["Mixed_6d"],
    ["Mixed_6e"],
    ["Mixed_7a"],
    ["Mixed_7b"],
    ["Mixed_7c"],
    ["avgpool"],
    ["dropout"]
]

predictedClass, probabilities = testExperiment( config, weights, studyID, layerMatrix )

print("Predicted class: ", getClassDescription(predictedClass, num_classes))
print("Confidence: ", probabilities[predictedClass].numpy())
print("Probs: ", probabilities.numpy())
