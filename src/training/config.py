"""
src/training/config.py
======================
Central configuration for the entire ISL Fingerspelling pipeline.

WHY A SINGLE CONFIG FILE?
--------------------------
Keeping all hyperparameters in one place prevents the common thesis mistake
of running experiments with different settings across scripts and then being
unable to reproduce a result. Every script imports from here.

DESIGN DECISION: Plain Python dataclasses instead of YAML/JSON.
Reason: type-checked, importable, no parsing overhead, and Python-native.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent.parent   # ISL fingerspelling/
DATA_DIR      = ROOT / "data" / "continious-Fingerspelling-dataset"
STATIC_DIR    = ROOT / "data" / "static-isl-dataset" / "isl_data"
VIDEOS_DIR    = DATA_DIR / "videos"
ANN_CSV       = DATA_DIR / "fingerspelling_annotations.csv"
LOC_CSV       = DATA_DIR / "localization_timestamps.csv"
SPLIT_CSV     = DATA_DIR / "split_info.csv"

CACHE_DIR     = ROOT / "cache"
CKPT_DIR      = ROOT / "checkpoints"
LOG_DIR       = ROOT / "logs"
RESULTS_DIR   = ROOT / "results"

FINAL_SPLIT_CSV = CACHE_DIR / "final_split.csv"
EXTRACTION_STATS_CSV = CACHE_DIR / "extraction_stats.csv"

# ─────────────────────────────────────────────────────────────────────────────
# VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────
# DESIGN DECISION (DD-001): 26 lowercase letters only.
# CTC blank token is the LAST index (26) — this is the PyTorch CTCLoss convention
# when zero_infinity=True and blank=26.
VOCAB = list("abcdefghijklmnopqrstuvwxyz")  # indices 0–25
BLANK_IDX = len(VOCAB)                       # 26
NUM_CLASSES = len(VOCAB) + 1                 # 27  (26 letters + blank)

CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}

# ─────────────────────────────────────────────────────────────────────────────
# DATASET / SPLIT
# ─────────────────────────────────────────────────────────────────────────────
VAL_SPLIT_RATIO   = 0.10   # 10% of train source-videos → val (DD-002)
RANDOM_SEED       = 42     # fixed for reproducibility

# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK EXTRACTION (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────
NUM_LANDMARKS     = 21
LANDMARK_DIM      = 3      # (x, y, z) per landmark
FEATURE_DIM       = NUM_LANDMARKS * LANDMARK_DIM  # 63

# Missing-detection handling (DD-003)
MAX_FORWARD_FILL_FRAMES = 5      # max consecutive frames to forward-fill
DROP_SEGMENT_THRESHOLD  = 0.50   # drop segment if > 50% frames are zero-filled

# MediaPipe settings
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE  = 0.5
MP_MAX_NUM_HANDS            = 1   # Used for continuous pipeline (dominant hand only)
MP_MAX_NUM_HANDS_STATIC     = 2   # Used for static pipeline (both hands for ISL)

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────
WRIST_IDX  = 0    # MediaPipe landmark 0 = wrist (translation reference)
SCALE_IDX  = 9    # MediaPipe landmark 9 = middle finger MCP (scale reference)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 32
NUM_WORKERS = 0    # Windows often has issues with >0; set to 2+ on Linux

# ─────────────────────────────────────────────────────────────────────────────
# STATIC ISL DATASET PATHS (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────
STATIC_DATA_DIR   = ROOT / "data" / "static-isl-dataset"
STATIC_ISL_DIR    = STATIC_DATA_DIR / "Static_ISL"        # ~1200 imgs/class, A-Z
KAGGLE_ISL_DIR    = STATIC_DATA_DIR / "Kaggle_ISL_Dataset" # ~25 imgs/class, 23 letters
ISL_DATA_DIR      = ROOT / "data" / "ISL_DATA"            # ~460-700 imgs/class, A-Z + 1-9
STATIC_CACHE_DIR  = ROOT / "cache" / "static"             # extracted landmarks

# ISL static letters (A-Z only — digits excluded for fingerspelling task)
STATIC_LETTERS    = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")   # 26 classes
NUM_STATIC_CLASSES = len(STATIC_LETTERS)                  # 26
LETTER_TO_IDX     = {ch: i for i, ch in enumerate(STATIC_LETTERS)}
IDX_TO_LETTER     = {i: ch for i, ch in enumerate(STATIC_LETTERS)}

# ISL Hand Skeleton: MediaPipe landmark connectivity (edge list for GAT)
# Each hand has 21 landmarks. Edges follow the finger bone structure.
# Reference: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
HAND_EDGES = [
    # Palm
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (5, 9), (9, 13), (13, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
]
NUM_HAND_EDGES = len(HAND_EDGES)  # 21 edges

# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM BASELINE (Stage 4)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BiLSTMConfig:
    """
    Hyperparameters for the BiLSTM + CTC baseline.
    WHY THESE VALUES: see DD-006 in DESIGN_DECISIONS.md
    """
    input_dim:    int   = FEATURE_DIM   # 63 (raw landmarks)
    proj_dim:     int   = 128           # linear projection before LSTM
    hidden_dim:   int   = 256           # LSTM hidden size per direction
    num_layers:   int   = 2             # number of BiLSTM layers
    dropout:      float = 0.3           # dropout between LSTM layers
    num_classes:  int   = NUM_CLASSES   # 27

    # Training
    learning_rate:    float = 1e-3
    weight_decay:     float = 1e-4
    max_epochs:       int   = 100
    patience:         int   = 10        # early stopping patience (epochs)
    lr_factor:        float = 0.5       # ReduceLROnPlateau factor
    lr_patience:      int   = 5         # ReduceLROnPlateau patience

# ─────────────────────────────────────────────────────────────────────────────
# CONFORMER (Stage 6)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ConformerConfig:
    """
    Hyperparameters for the Conformer + CTC main model.
    WHY THESE VALUES: see DD-007 in DESIGN_DECISIONS.md
    """
    input_dim:    int   = FEATURE_DIM   # 63
    d_model:      int   = 144           # must be divisible by num_heads
    num_heads:    int   = 4
    num_layers:   int   = 4
    conv_kernel:  int   = 31            # ~1 second at 30 FPS
    ff_expansion: int   = 4
    dropout:      float = 0.1
    num_classes:  int   = NUM_CLASSES   # 27

    # Training
    learning_rate:    float = 5e-4      # slightly lower than BiLSTM (larger model)
    weight_decay:     float = 1e-4
    max_epochs:       int   = 100
    patience:         int   = 10
    lr_factor:        float = 0.5
    lr_patience:      int   = 5

# ─────────────────────────────────────────────────────────────────────────────
# DH-GAT STATIC CLASSIFIER (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StaticConfig:
    """
    Hyperparameters for the Dual-Hand Graph Attention Transformer (DH-GAT).

    WHY THIS ARCHITECTURE:
    ISL fingerspelling is a BSL-derived TWO-HANDED system. Each hand's 21
    landmarks form a graph (nodes = joints, edges = finger bones). We use
    Graph Attention to respect this structure, then fuse both hand embeddings.

    INPUT:  right_node_feat (21,6) + left_node_feat (21,6)
            + right_global_geo (10,) + left_global_geo (10,)
            + presence_flags (2,) = enriched dual-hand representation
    OUTPUT: 26-class softmax (A-Z)
    """
    # Per-hand GAT encoder
    in_features:    int   = 6             # enriched node features: (x,y,z, dist_wrist, angle, dist_tip)
    gat_hidden:     int   = 64            # hidden dim per GAT layer
    gat_heads:      int   = 4             # multi-head attention in GAT
    gat_layers:     int   = 2             # number of GAT layers per hand
    hand_embed_dim: int   = 128           # output embedding per hand

    # Global geometric features (inter-fingertip distances)
    global_geo_dim: int   = 10            # per-hand global geometric features

    # Fusion + classifier
    fusion_dim:     int   = 256           # fused representation dim
    dropout:        float = 0.3

    # Classes
    num_classes:    int   = NUM_STATIC_CLASSES  # 26

    # Training
    learning_rate:      float = 1e-3
    weight_decay:       float = 1e-4
    max_epochs:         int   = 80
    patience:           int   = 15        # static trains faster, more patience
    batch_size:         int   = 64
    label_smoothing:    float = 0.1       # regularization

    # Webcam inference
    stable_frames:      int   = 10        # frames prediction must be stable before display
    min_confidence:     float = 0.70      # minimum softmax confidence to display

# Default instances
bilstm_cfg    = BiLSTMConfig()
conformer_cfg = ConformerConfig()
static_cfg    = StaticConfig()


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────
WRIST_IDX  = 0    # MediaPipe landmark 0 = wrist (translation reference)
SCALE_IDX  = 9    # MediaPipe landmark 9 = middle finger MCP (scale reference)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 32
NUM_WORKERS = 0    # Windows often has issues with >0; set to 2+ on Linux

# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM BASELINE (Stage 4)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BiLSTMConfig:
    """
    Hyperparameters for the BiLSTM + CTC baseline.
    WHY THESE VALUES: see DD-006 in DESIGN_DECISIONS.md
    """
    input_dim:    int   = FEATURE_DIM   # 63 (raw landmarks)
    proj_dim:     int   = 128           # linear projection before LSTM
    hidden_dim:   int   = 256           # LSTM hidden size per direction
    num_layers:   int   = 2             # number of BiLSTM layers
    dropout:      float = 0.3           # dropout between LSTM layers
    num_classes:  int   = NUM_CLASSES   # 27

    # Training
    learning_rate:    float = 1e-3
    weight_decay:     float = 1e-4
    max_epochs:       int   = 100
    patience:         int   = 10        # early stopping patience (epochs)
    lr_factor:        float = 0.5       # ReduceLROnPlateau factor
    lr_patience:      int   = 5         # ReduceLROnPlateau patience

# ─────────────────────────────────────────────────────────────────────────────
# CONFORMER (Stage 6)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ConformerConfig:
    """
    Hyperparameters for the Conformer + CTC main model.
    WHY THESE VALUES: see DD-007 in DESIGN_DECISIONS.md
    """
    input_dim:    int   = FEATURE_DIM   # 63
    d_model:      int   = 144           # must be divisible by num_heads
    num_heads:    int   = 4
    num_layers:   int   = 4
    conv_kernel:  int   = 31            # ~1 second at 30 FPS
    ff_expansion: int   = 4
    dropout:      float = 0.1
    num_classes:  int   = NUM_CLASSES   # 27

    # Training
    learning_rate:    float = 5e-4      # slightly lower than BiLSTM (larger model)
    weight_decay:     float = 1e-4
    max_epochs:       int   = 100
    patience:         int   = 10
    lr_factor:        float = 0.5
    lr_patience:      int   = 5

# Default instances
bilstm_cfg    = BiLSTMConfig()
conformer_cfg = ConformerConfig()
