import os
import torch
import random
import numpy as np

try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except:
    BASE_DIR = os.getcwd()

DATA_RAW  = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
SPLITS    = os.path.join(BASE_DIR, "data", "splits")

CKPT_SEG   = os.path.join(BASE_DIR, "outputs", "checkpoints", "segmentation")
CKPT_CLS   = os.path.join(BASE_DIR, "outputs", "checkpoints", "classification")
CKPT_MULTI = os.path.join(BASE_DIR, "outputs", "checkpoints", "multitask")

LOG_SEG    = os.path.join(BASE_DIR, "outputs", "logs", "segmentation")
LOG_CLS    = os.path.join(BASE_DIR, "outputs", "logs", "classification")
LOG_MULTI  = os.path.join(BASE_DIR, "outputs", "logs", "multitask")

for path in [
    DATA_PROC, SPLITS,
    CKPT_SEG, CKPT_CLS, CKPT_MULTI,
    LOG_SEG, LOG_CLS, LOG_MULTI
]:
    os.makedirs(path, exist_ok=True)

CENTERS = ["Center_1", "Center_2", "Center_3", "Center_4"]

PATCH_SEG     = 160
PATCH_CLS     = 224
EXPAND_VOXEL  = 5
RANDOM_OFFSET = 12
RANDOM_SEED   = 42

TEST_RATIO_SEG = 0.25
TEST_RATIO_CLS = 0.30
VAL_RATIO      = 0.15

CV_FOLDS = 5

SEG_EPOCHS  = 180
SEG_BATCH   = 8
SEG_LR      = 3e-4
SEG_IN_CH   = 1
SEG_OUT_CH  = 1

CLS_EPOCHS       = 200
CLS_BATCH        = 32
CLS_LR_HEAD      = 3e-4
CLS_LR_FINETUNE  = 1e-5
CLS_FINETUNE_EP  = 40
CLS_CLASSES      = 2
CLS_PRETRAINED   = True

DICE_SMOOTH  = 1.0
FOCAL_GAMMA  = 2.0
FOCAL_ALPHA  = 0.25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 0
PIN_MEMORY  = True

SEG_THRESHOLD = 0.5
SAVE_EVERY    = 50
LOG_EVERY     = 10
PATIENCE      = 30

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False