"""Single place for paths and data-prep settings.

Jupyter on this PC uses the project folder. Colab uses MyDrive when that
folder exists. The SDFVD zip must sit at DRIVE_ROOT/sdfvd.zip.
Every other path is derived from DRIVE_ROOT and OUTPUT_DIR.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_PACKAGE_DIR)

# Colab: MyDrive after drive.mount. Jupyter: this project folder.
if os.path.isdir("/content/drive/MyDrive"):
    _DEFAULT_ROOT = "/content/drive/MyDrive"
else:
    _DEFAULT_ROOT = _PROJECT_DIR

# The SDFVD zip must sit at DRIVE_ROOT/sdfvd.zip.
DRIVE_ROOT = os.environ.get("DRIVE_ROOT", _DEFAULT_ROOT)

# Cached faces, manifests, videos, and checkpoints.
# This PC's C: drive is tight, so local runs write to D: when that drive exists.
if os.path.isdir("/content/drive/MyDrive"):
    _DEFAULT_OUTPUT = os.path.join(_DEFAULT_ROOT, "deepfake_phase2")
elif os.path.isdir("D:\\"):
    _DEFAULT_OUTPUT = r"D:\deepfake_phase2"
else:
    _DEFAULT_OUTPUT = os.path.join(_PROJECT_DIR, "deepfake_phase2")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", _DEFAULT_OUTPUT)

# Already on Drive. The sdfvd step unzips this; it does not download it.
SDFVD_ZIP = os.path.join(DRIVE_ROOT, "sdfvd.zip")

# Hugging Face cache lives under the output dir so a Colab restart can resume.
HF_HOME = os.path.join(OUTPUT_DIR, "hf_home")

# Folder names created under OUTPUT_DIR.
DIR_INSPECT = "inspect"
DIR_MANIFESTS = "manifests"
DIR_LOGS = "logs"
DIR_FFPP_RAW_TEST = "ffpp_raw_test"  # pre-crop PNGs, test videos only (for c40)
DIR_FFPP_FACES = "ffpp_faces"
DIR_FFPP_FREQ = "ffpp_freq"
DIR_C40_VIDEOS = "c40_style_videos"
DIR_C40_FRAMES = "c40_style_decoded"  # frames read back from the crf-40 mp4
DIR_C40_FACES = "c40_style_faces"
DIR_C40_FREQ = "c40_style_freq"
DIR_SDFVD_UNZIP = "sdfvd_unzipped"
DIR_SDFVD_FACES = "sdfvd_faces"
DIR_SDFVD_FREQ = "sdfvd_freq"
DIR_CHECKPOINTS = "checkpoints"
DIR_RESULTS = "results"

# ---------------------------------------------------------------------------
# FaceForensics++ mirror (Hugging Face)
# ---------------------------------------------------------------------------

# Third-party frame mirror. Frames are already extracted, so the prep script
# keeps every provided frame_idx. It does not sample the original videos.
HF_DATASET_ID = "TsienDragon/ffplusplus_c23_frames"

# The mirror also has a "val" split (~5.3k rows). The spec pools train and test
# and then re-splits by video. Change this tuple to ("train", "val", "test")
# if every real video in the mirror should be eligible.
POOL_SPLITS = ("train", "test")

REAL_CATEGORY = "original"
FAKE_CATEGORIES = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
KEEP_CATEGORIES = (REAL_CATEGORY,) + FAKE_CATEGORIES
DROP_CATEGORIES = ("FaceShifter", "DeepFakeDetection")

# A video must have at least this many mirror frames before it can be selected.
# Sequence building runs later, after faces with no detection are dropped.
MIN_FRAMES_PER_VIDEO = 10

# "auto" decides from the saved sample image. Set "full_frame" or "face_crop"
# to override that decision after you look at inspect/sample_frame.jpg.
FACE_INPUT_MODE = "auto"

# ---------------------------------------------------------------------------
# Faces, sequences, frequency
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224
# Pixels added on each side of the MTCNN box before the square crop.
# Landmarks are ignored: there is no alignment warp.
MTCNN_MARGIN = 20
MTCNN_MIN_FACE_SIZE = 40
MTCNN_KEEP_PROB = 0.90

# 10 consecutive kept frames, stride 10: windows inside one video do not overlap.
SEQ_LEN = 10
SEQ_STRIDE = 10
MAX_SEQS_PER_VIDEO = 6

# ImageNet RGB normalization. Applied when a batch is loaded for training,
# not baked into the saved jpgs.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Video-level split
# ---------------------------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

# How videos are grouped before the 70/15/15 split.
# "all_persons": every numeric id in a name such as 000_003 stays in one split,
#   so the same person is not in two splits.
# "source": group only by the first id (000 from 000_003). Use this if
#   all_persons ties the subset into one component and three splits are impossible.
IDENTITY_GROUPING = "all_persons"

# Binary labels written into manifests. Fake is the positive class.
LABEL_REAL = 0
LABEL_FAKE = 1

# ---------------------------------------------------------------------------
# SDFVD (test only, no training)
# ---------------------------------------------------------------------------

SDFVD_FRAME_EVERY = 5
SDFVD_MAX_FRAMES = 60
# Folder-name match (full path component, lowercased).
SDFVD_REAL_DIR_NAMES = ("videos_real", "real", "original", "originals", "pexels")
SDFVD_FAKE_DIR_NAMES = ("videos_fake", "fake", "fakes", "deepfake", "deepfakes", "remaker")
SDFVD_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# ---------------------------------------------------------------------------
# Compression robustness: c40-style (simulated)
# ---------------------------------------------------------------------------

# This is a libx264 CRF-40 re-encode of our FF++ test frames.
# It is not the official FaceForensics++ c40 release.
C40_LABEL = "c40-style (simulated)"
C40_CRF = 40
C40_FPS = 10  # container timing only; frames are already sparse

# ---------------------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------------------

# Switch one variant on for a single run. `train --all` still trains every
# variant below, then evaluates each test set once.
# spatial | spatial_freq | spatial_temporal | full
# "spatial" is the spatial-only baseline.
MODEL_VARIANT = "full"
ALL_VARIANTS = ("spatial", "spatial_freq", "spatial_temporal", "full")

# Frequency CNN width, Bi-LSTM hidden size, and the shared classification head.
FREQ_OUT_DIM = 128
LSTM_HIDDEN = 256
HEAD_HIDDEN = 256
DROPOUT = 0.3

BATCH_SIZE = 8
# Optimizer still steps every 8 sequences. This 4GB RTX 3050 fits 4 clips
# at a time (measured peak about 1.9GB), so two micro-batches make one step.
MICRO_BATCH_SIZE = 4
MAX_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LR_FACTOR = 0.5
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE = 3
LR_PLATEAU_PATIENCE = 1
# Sequence score at or above this value counts as fake.
DECISION_THRESHOLD = 0.5
# Windows Jupyter hangs if DataLoader spawns worker processes.
NUM_WORKERS = 0 if os.name == "nt" else 2
