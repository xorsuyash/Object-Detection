import torch 
import os 

BASE_PATH="dataset"
IMAGES_PATH=os.path.sep.join([BASE_PATH,"images"])
ANNOTS_PATH=os.path.sep.join([BASE_PATH,"annotations"])


BASE_OUTPUT="output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pth"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


lr=1e-4
batch_size=32
num_epochs=20

labels=1.0
bbox=1.0


