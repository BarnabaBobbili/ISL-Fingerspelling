"""
src/preprocessing/feature_engineer.py
======================================
Geometric feature engineering for ISL hand landmarks.

WHY GEOMETRIC FEATURES:
  Raw (x,y,z) coordinates are sensitive to:
    - Camera angle / perspective projection
    - Hand distance from camera
    - Wrist rotation (3D orientation)
  Geometric features (inter-landmark distances, joint angles) are invariant to
  rigid-body transformations — they depend only on the SHAPE of the hand,
  not where or how it is positioned relative to the camera.
  This directly bridges the domain gap between curated dataset images and
  real-time webcam captures.

FEATURE DESIGN:
  Per-node features (21 nodes × 6 features = 126):
    - (x, y, z)                          : 3 — position in normalized space
    - distance to wrist (landmark 0)     : 1 — finger extension indicator
    - angle at this joint                : 1 — bend indicator (0 if not a joint)
    - distance to fingertip of same finger: 1 — curl depth indicator

  Global geometric features (10 per hand):
    - All 10 pairwise inter-fingertip distances
      {thumb(4), index(8), middle(12), ring(16), pinky(20)}
    - These capture finger spread patterns, key for V/U/R, M/N/W distinctions

MediaPipe landmark indices:
  0=wrist, 1=thumb_cmc, 2=thumb_mcp, 3=thumb_ip, 4=thumb_tip
  5=index_mcp, 6=index_pip, 7=index_dip, 8=index_tip
  9=middle_mcp, 10=middle_pip, 11=middle_dip, 12=middle_tip
  13=ring_mcp, 14=ring_pip, 15=ring_dip, 16=ring_tip
  17=pinky_mcp, 18=pinky_pip, 19=pinky_dip, 20=pinky_tip

Run self-test:
  python src/preprocessing/feature_engineer.py --self-test
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Fingertip indices
FINGERTIPS = [4, 8, 12, 16, 20]

# All 10 pairs of fingertips (combinatorial)
FINGERTIP_PAIRS = [
    (4,  8),   # thumb  ↔ index
    (4,  12),  # thumb  ↔ middle
    (4,  16),  # thumb  ↔ ring
    (4,  20),  # thumb  ↔ pinky
    (8,  12),  # index  ↔ middle
    (8,  16),  # index  ↔ ring
    (8,  20),  # index  ↔ pinky
    (12, 16),  # middle ↔ ring
    (12, 20),  # middle ↔ pinky
    (16, 20),  # ring   ↔ pinky
]

# Fingertip-to-MCP pairs (curl depth: tip to its own MCP base)
CURL_PAIRS = [
    (4,  1),   # thumb  tip  → thumb  CMC
    (8,  5),   # index  tip  → index  MCP
    (12, 9),   # middle tip  → middle MCP
    (16, 13),  # ring   tip  → ring   MCP
    (20, 17),  # pinky  tip  → pinky  MCP
]

# Per-finger joint triplets for angle computation: (A, joint_B, C)
# angle at B = angle(A, B, C) = arccos( (BA · BC) / (|BA| * |BC|) )
JOINT_TRIPLETS = [
    # Thumb
    (0,  1,  2),   # thumb MCP  angle
    (1,  2,  3),   # thumb IP   angle (between CMC-MCP and MCP-IP)
    (2,  3,  4),   # thumb DIP  angle (tip bend)
    # Index
    (0,  5,  6),   # index MCP  angle
    (5,  6,  7),   # index PIP  angle
    (6,  7,  8),   # index DIP  angle
    # Middle
    (0,  9,  10),  # middle MCP angle
    (9,  10, 11),  # middle PIP angle
    (10, 11, 12),  # middle DIP angle
    # Ring
    (0,  13, 14),  # ring MCP   angle
    (13, 14, 15),  # ring PIP   angle
    (14, 15, 16),  # ring DIP   angle
    # Pinky
    (0,  17, 18),  # pinky MCP  angle
    (17, 18, 19),  # pinky PIP  angle
    (18, 19, 20),  # pinky DIP  angle
]

# Which landmark is the "joint center" in each triplet (index into JOINT_TRIPLETS)
# Maps landmark index → pre-computed angle index (-1 if not a joint center)
_JOINT_CENTER_TO_ANGLE_IDX = {}
for _angle_idx, (_, _joint, _) in enumerate(JOINT_TRIPLETS):
    _JOINT_CENTER_TO_ANGLE_IDX[_joint] = _angle_idx

# Maps each landmark to the fingertip of its finger
# 0=wrist → no fingertip (use 0), otherwise map to the fingertip of that chain
_LM_TO_FINGERTIP = {
    0:  0,   # wrist — no natural fingertip, use self (dist will be 0 after subtraction)
    1:  4,   # thumb  chain
    2:  4,
    3:  4,
    4:  4,
    5:  8,   # index  chain
    6:  8,
    7:  8,
    8:  8,
    9:  12,  # middle chain
    10: 12,
    11: 12,
    12: 12,
    13: 16,  # ring   chain
    14: 16,
    15: 16,
    16: 16,
    17: 20,  # pinky  chain
    18: 20,
    19: 20,
    20: 20,
}

NUM_GLOBAL_DIST = len(FINGERTIP_PAIRS)   # 10
NUM_NODE_FEATURES = 6                    # (x,y,z, dist_wrist, angle, dist_tip)
NUM_JOINT_ANGLES = len(JOINT_TRIPLETS)   # 15


# ─────────────────────────────────────────────────────────────────────────────
# Core computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _three_point_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the angle at joint B formed by rays B→A and B→C.
    Returns value in [0, 1] (normalized by π).
    Handles degenerate cases (zero-length vectors) gracefully.
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)   # numerical safety
    return float(np.arccos(cos_angle) / np.pi)   # normalize to [0, 1]


def _compute_all_angles(lm: np.ndarray) -> np.ndarray:
    """
    Compute all 15 joint angles for a (21, 3) landmark array.
    Returns (15,) array, each value in [0, 1].
    """
    angles = np.zeros(NUM_JOINT_ANGLES, dtype=np.float32)
    for i, (a_idx, b_idx, c_idx) in enumerate(JOINT_TRIPLETS):
        angles[i] = _three_point_angle(lm[a_idx], lm[b_idx], lm[c_idx])
    return angles


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_node_features(lm: np.ndarray) -> np.ndarray:
    """
    Compute enriched per-node feature matrix from a single (21, 3) hand landmark
    array (must already be wrist-centric normalized).

    Each of the 21 nodes gets 6 features:
      [x, y, z, dist_to_wrist, angle_at_joint, dist_to_fingertip]

    Parameters
    ----------
    lm : (21, 3) float32, wrist-centric normalized landmarks

    Returns
    -------
    (21, 6) float32 — enriched per-node feature matrix
    """
    assert lm.shape == (21, 3), f"Expected (21,3), got {lm.shape}"
    lm = lm.astype(np.float32)

    # Pre-compute all angles (15,)
    all_angles = _compute_all_angles(lm)

    node_feat = np.zeros((21, NUM_NODE_FEATURES), dtype=np.float32)

    for i in range(21):
        # Features 0-2: raw (x, y, z)
        node_feat[i, 0:3] = lm[i]

        # Feature 3: Euclidean distance to wrist (landmark 0)
        node_feat[i, 3] = float(np.linalg.norm(lm[i] - lm[0]))

        # Feature 4: joint angle at this landmark (0 if not a joint center)
        if i in _JOINT_CENTER_TO_ANGLE_IDX:
            node_feat[i, 4] = all_angles[_JOINT_CENTER_TO_ANGLE_IDX[i]]
        else:
            node_feat[i, 4] = 0.0

        # Feature 5: distance from this node to the fingertip of its finger
        tip_idx = _LM_TO_FINGERTIP[i]
        node_feat[i, 5] = float(np.linalg.norm(lm[i] - lm[tip_idx]))

    return node_feat


def compute_global_distances(lm: np.ndarray) -> np.ndarray:
    """
    Compute the 10 pairwise inter-fingertip distances for a single hand.
    These are rotation/translation/scale-invariant features capturing finger spread.

    Parameters
    ----------
    lm : (21, 3) float32, wrist-centric normalized landmarks

    Returns
    -------
    (10,) float32 — inter-fingertip distances
    """
    assert lm.shape == (21, 3), f"Expected (21,3), got {lm.shape}"
    lm = lm.astype(np.float32)

    dists = np.zeros(NUM_GLOBAL_DIST, dtype=np.float32)
    for i, (a, b) in enumerate(FINGERTIP_PAIRS):
        dists[i] = float(np.linalg.norm(lm[a] - lm[b]))
    return dists


def compute_joint_angles(lm: np.ndarray) -> np.ndarray:
    """
    Compute all 15 joint angles for a hand.
    Useful standalone (e.g., for the GeoMLP model).

    Parameters
    ----------
    lm : (21, 3) float32, wrist-centric normalized

    Returns
    -------
    (15,) float32 — joint angles normalized to [0, 1]
    """
    assert lm.shape == (21, 3), f"Expected (21,3), got {lm.shape}"
    return _compute_all_angles(lm.astype(np.float32))


def compute_fingertip_to_wrist_distances(lm: np.ndarray) -> np.ndarray:
    """
    Compute the 5 fingertip-to-wrist distances (finger extension measure).

    Parameters
    ----------
    lm : (21, 3) float32, wrist-centric normalized

    Returns
    -------
    (5,) float32
    """
    lm = lm.astype(np.float32)
    return np.array([np.linalg.norm(lm[tip]) for tip in FINGERTIPS], dtype=np.float32)


def compute_curl_distances(lm: np.ndarray) -> np.ndarray:
    """
    Compute the 5 fingertip-to-MCP distances (finger curl depth).

    Parameters
    ----------
    lm : (21, 3) float32, wrist-centric normalized

    Returns
    -------
    (5,) float32
    """
    lm = lm.astype(np.float32)
    return np.array([np.linalg.norm(lm[tip] - lm[mcp]) for tip, mcp in CURL_PAIRS],
                    dtype=np.float32)


def compute_geo_vector(lm: np.ndarray) -> np.ndarray:
    """
    Compute the full 35-dimensional geometric feature vector for one hand.
    Used by the GeoMLP model (no raw coordinates, pure geometry).

      10 inter-fingertip distances
    +  5 fingertip-to-wrist distances
    +  5 fingertip-to-MCP (curl) distances
    + 15 joint angles
    = 35 features total

    Parameters
    ----------
    lm : (21, 3) float32, wrist-centric normalized

    Returns
    -------
    (35,) float32
    """
    lm = lm.astype(np.float32)
    return np.concatenate([
        compute_global_distances(lm),             # 10
        compute_fingertip_to_wrist_distances(lm), #  5
        compute_curl_distances(lm),               #  5
        compute_joint_angles(lm),                 # 15
    ]).astype(np.float32)


def compute_contact_features(lm: np.ndarray, sigma: float = 0.12) -> np.ndarray:
    """
    Compute soft fingertip contact scores for all 10 fingertip pairs.

    WHY: Many ISL signs are defined by which fingertips are touching:
      - O / OK: all fingertips approach the thumb
      - F: index tip touches thumb tip
      - A: fingertips curl to palm (all tip-wrist distances small)
      - D / P: specific finger-to-thumb contact
    A Gaussian score exp(-d²/σ²) gives 1.0 when tips touch, decays to 0 as
    they separate, and is differentiable (no hard threshold needed).

    Parameters
    ----------
    lm    : (21, 3) float32, wrist-centric normalized
    sigma : contact radius in normalized hand space (default 0.12)

    Returns
    -------
    (10,) float32 — contact scores in [0, 1], ordered as FINGERTIP_PAIRS
    """
    lm = lm.astype(np.float32)
    contacts = np.zeros(len(FINGERTIP_PAIRS), dtype=np.float32)
    for i, (a, b) in enumerate(FINGERTIP_PAIRS):
        d = float(np.linalg.norm(lm[a] - lm[b]))
        contacts[i] = float(np.exp(-(d ** 2) / (sigma ** 2)))
    return contacts


def compute_palm_normal(lm: np.ndarray) -> np.ndarray:
    """
    Compute the unit normal vector to the palm plane.

    WHY: Signs D and P differ mainly in palm orientation (D faces outward,
    P faces downward/inward). The palm normal captures this orientation
    independently of hand position and scale — directly addressing the D/P
    confusion caused by orientation-flipped signs.

    The palm plane is defined by three stable, non-collinear landmarks:
      - Wrist (0)
      - Index MCP (5)
      - Pinky MCP (17)
    Normal = (index_MCP - wrist) × (pinky_MCP - wrist), L2-normalized.

    Parameters
    ----------
    lm : (21, 3) float32, wrist-centric normalized

    Returns
    -------
    (3,) float32 — unit palm normal vector, or zeros if degenerate
    """
    lm = lm.astype(np.float32)
    v1 = lm[5]  - lm[0]   # wrist → index MCP
    v2 = lm[17] - lm[0]   # wrist → pinky MCP
    normal = np.cross(v1, v2).astype(np.float32)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)   # degenerate / absent hand
    return (normal / norm).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
def _self_test():
    print("Running feature_engineer self-tests...")

    # Reproducible random landmarks (wrist-centric: landmark 0 should be ~zero)
    rng = np.random.default_rng(42)
    lm_raw = rng.standard_normal((21, 3)).astype(np.float32)
    lm_raw -= lm_raw[0]  # wrist-centric

    # --- compute_node_features ---
    nf = compute_node_features(lm_raw)
    assert nf.shape == (21, 6), f"Expected (21,6), got {nf.shape}"
    # First 3 features must match raw landmarks
    np.testing.assert_allclose(nf[:, :3], lm_raw, atol=1e-6)
    # Wrist node (index 0): dist_to_wrist == 0
    assert abs(nf[0, 3]) < 1e-6, f"Wrist dist_to_wrist should be 0, got {nf[0, 3]}"
    # All angles in [0, 1]
    assert nf[:, 4].min() >= 0.0 and nf[:, 4].max() <= 1.0, "Angles out of [0,1]"
    print(f"  compute_node_features: shape={nf.shape} OK")

    # --- compute_global_distances ---
    gd = compute_global_distances(lm_raw)
    assert gd.shape == (10,), f"Expected (10,), got {gd.shape}"
    assert (gd >= 0).all(), "Distances must be non-negative"
    print(f"  compute_global_distances: shape={gd.shape} OK")

    # --- compute_joint_angles ---
    angles = compute_joint_angles(lm_raw)
    assert angles.shape == (15,), f"Expected (15,), got {angles.shape}"
    assert angles.min() >= 0.0 and angles.max() <= 1.0, "Angles out of [0,1]"
    print(f"  compute_joint_angles: shape={angles.shape} OK")

    # --- compute_geo_vector ---
    gv = compute_geo_vector(lm_raw)
    assert gv.shape == (35,), f"Expected (35,), got {gv.shape}"
    print(f"  compute_geo_vector: shape={gv.shape} OK")

    # --- compute_contact_features ---
    cf = compute_contact_features(lm_raw)
    assert cf.shape == (10,), f"Expected (10,), got {cf.shape}"
    assert cf.min() >= 0.0 and cf.max() <= 1.0, "Contact scores must be in [0,1]"
    # Zero hand should give near-1 scores (all tips at origin = touching)
    cf_zero = compute_contact_features(np.zeros((21, 3), dtype=np.float32))
    assert (cf_zero > 0.99).all(), "All-zero hand should give contact ≈ 1.0"
    print(f"  compute_contact_features: shape={cf.shape} OK")

    # --- compute_palm_normal ---
    pn = compute_palm_normal(lm_raw)
    assert pn.shape == (3,), f"Expected (3,), got {pn.shape}"
    norm_val = float(np.linalg.norm(pn))
    assert abs(norm_val - 1.0) < 1e-5, f"Palm normal must be unit vector, got norm={norm_val}"
    # Zero hand should return zeros (graceful degenerate case)
    pn_zero = compute_palm_normal(np.zeros((21, 3), dtype=np.float32))
    assert (pn_zero == 0).all(), "Zero hand should give zero palm normal"
    print(f"  compute_palm_normal: shape={pn.shape}, norm={norm_val:.6f} OK")

    # --- Invariance check: translation should not change geometric features ---
    lm_shifted = lm_raw.copy()
    lm_shifted += np.array([0.5, 0.3, -0.2], dtype=np.float32)  # translate all joints
    # Re-normalize (wrist-centric)
    lm_shifted -= lm_shifted[0]
    gd_shifted = compute_global_distances(lm_shifted)
    np.testing.assert_allclose(gd, gd_shifted, atol=1e-5,
                               err_msg="Global distances should be translation-invariant after normalization")
    print("  Translation invariance: OK")

    # --- Zero hand (absent) should produce all-zero features ---
    lm_zero = np.zeros((21, 3), dtype=np.float32)
    nf_zero = compute_node_features(lm_zero)
    assert (nf_zero == 0).all(), "Zero landmarks should give all-zero node features"
    gv_zero = compute_geo_vector(lm_zero)
    assert (gv_zero == 0).all(), "Zero landmarks should give all-zero geo vector"
    print("  Zero (absent) hand: OK")

    print("\nAll feature_engineer self-tests PASSED\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
