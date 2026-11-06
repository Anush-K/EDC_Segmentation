"""Shared helpers. Python 3.8 compatible."""
import os, json, random
import numpy as np

DATASETS = ["busi", "lgg", "covid19", "kvasir"]


def seed_everything(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    import torch
    # each DataLoader worker gets its own, reproducible numpy/random seed
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s + worker_id); random.seed(s + worker_id)


def lgg_patient(name):
    """TCGA_CS_4941_19960909_11.tif -> TCGA_CS_4941 (first three '_' parts)."""
    return "_".join(os.path.basename(name).split("_")[:3])


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)