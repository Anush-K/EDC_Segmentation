import os
import torch

try:
    from IPython import get_ipython
    ipython = get_ipython()
    IN_COLAB = ipython is not None and "google.colab" in str(ipython)
except ImportError:
    IN_COLAB = False

ENV = "colab" if IN_COLAB else "local"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#BASE_DIR = "/home/cs24d0008/EDC_SSL"
BASE_DIR = "/Users/anushk/Desktop/EDC_Segmentation"

CODE_DIR = os.path.join(BASE_DIR, "EDC-master")

DATASET_DIR = os.path.join(BASE_DIR, "LGG")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR  = os.path.join(DATASET_DIR, "test")

SAVED_MODELS_DIR = os.path.join(CODE_DIR, "saved_models")


# --------------------------------------------------
# Pretty print configuration
# --------------------------------------------------
def print_config():
    print("===== LGG CONFIGURATION =====")
    print(f"Environment:     {ENV}")
    print(f"Device:          {device}")
    print(f"Base directory:  {BASE_DIR}")
    print(f"Code directory:  {CODE_DIR}")
    print(f"Dataset root:    {DATASET_DIR}")
    print(f"Train folder:    {TRAIN_DIR}")
    print(f"Test folder:     {TEST_DIR}")
    print(f"Saved models:    {SAVED_MODELS_DIR}")
    print("================================")
