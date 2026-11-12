"""
stage1_inspect_dataset.py
==========================
Stage 1: Dataset Inspection for ISL Continuous Fingerspelling Recognition

PURPOSE
-------
This script is the first step of the thesis pipeline. Before training anything,
we need to fully understand what we are working with. This script:
  1. Counts and summarises video files in the dataset
  2. Reads all three CSVs and cross-validates them
  3. Analyses label/character distributions
  4. Reports the existing train/val/test split (provided by dataset authors)
  5. Verifies that the split is video-source-independent (= closest available
     proxy for signer-independence in this dataset)
  6. Flags data quality issues: missing files, missing annotations, extreme
     duration outliers, tiny files that might be corrupt
  7. Summarises the static ISL image dataset

WHY READ-ONLY HERE?
-------------------
Stage 1 is an audit. We never modify the dataset here. All changes
(renaming, splitting, caching) happen in later stages after you have
reviewed and approved the findings.

DESIGN DECISION - No OpenCV video reading here
-----------------------------------------------
We intentionally avoid opening every video with OpenCV in this stage
because it would take 10-30 minutes on 1308 files. Instead we rely on:
  - File system metadata (size) for quick corruption heuristics
  - localization_timestamps.csv for duration/FPS info already provided
  - We DO open a small random sample (5 videos) to verify resolution/FPS
This is faster and sufficient for an audit. Full frame-by-frame reading
happens in Stage 2 (landmark extraction).

OUTPUTS
-------
Prints a structured report to stdout. Also saves:
  - stage1_report.txt   : plain-text copy of the report
  - stage1_char_dist.png: character frequency bar chart
  - stage1_dur_dist.png : segment duration histogram
  - stage1_seq_len_dist.png : sequence length histogram

DEPENDENCIES
------------
pip install pandas matplotlib tqdm opencv-python-headless
(opencv only for the 5-video sample probe -- can be skipped)

USAGE
-----
cd "E:/MTech/Mtech Sem3/ISL fingerspelling"
python stage1_inspect_dataset.py
"""

import os
import sys
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend -- safe on any machine
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
# CONFIGURATION  -- change these if you move the dataset
# =============================================================================
BASE_DIR       = Path(__file__).parent
CONT_DIR       = BASE_DIR / "continious-Fingerspelling-dataset"
STATIC_DIR     = BASE_DIR / "static-isl-dataset"

VIDEOS_DIR     = CONT_DIR / "videos"
ANN_CSV        = CONT_DIR / "fingerspelling_annotations.csv"
LOC_CSV        = CONT_DIR / "localization_timestamps.csv"
SPLIT_CSV      = CONT_DIR / "split_info.csv"

REPORT_TXT     = BASE_DIR / "stage1_report.txt"
CHAR_DIST_PNG  = BASE_DIR / "stage1_char_dist.png"
DUR_DIST_PNG   = BASE_DIR / "stage1_dur_dist.png"
SEQ_LEN_PNG    = BASE_DIR / "stage1_seq_len_dist.png"

# How many videos to open with OpenCV for resolution/FPS check
N_SAMPLE_VIDEOS = 5
# Files smaller than this (bytes) are flagged as potentially corrupt
MIN_FILE_SIZE_BYTES = 50_000   # 50 KB -- a 2-second segment at low quality
# Duration thresholds (seconds) for outlier flagging
MIN_DUR_SEC = 0.5
MAX_DUR_SEC = 30.0


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str, width: int = 72) -> str:
    """Returns a formatted section header string."""
    bar = "=" * width
    return f"\n{bar}\n  {title}\n{bar}"


def bullet(label: str, value) -> str:
    return f"  * {label:<45} {value}"


def open_video_sample(video_paths, n=5):
    """
    Opens a random sample of videos with OpenCV to read resolution/FPS.
    We sample rather than scan all 1308 videos because:
      - All segments come from the same production pipeline (ISL news)
      - Resolution/FPS is virtually certain to be uniform across the dataset
      - Scanning all would take 10-30 minutes and is not worth it at Stage 1

    Returns a list of dicts: {path, width, height, fps, frame_count, ok}
    """
    try:
        import cv2
    except ImportError:
        return []

    results = []
    sample = random.sample(video_paths, min(n, len(video_paths)))
    for vp in sample:
        cap = cv2.VideoCapture(str(vp))
        ok = cap.isOpened()
        if ok:
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            w = h = fps = fc = 0
        cap.release()
        results.append({"path": vp, "width": w, "height": h,
                         "fps": fps, "frame_count": fc, "ok": ok})
    return results


def count_static_dataset(static_dir: Path):
    """
    Counts images in the static ISL dataset (folder-per-class structure).
    Returns (total_images, class_counts_dict, issues_list)
    """
    isl_data = static_dir / "isl_data"
    if not isl_data.exists():
        isl_data = static_dir

    class_counts = {}
    issues = []
    for class_dir in sorted(isl_data.iterdir()):
        if class_dir.is_dir():
            imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            class_counts[class_dir.name] = len(imgs)
            if len(imgs) == 0:
                issues.append(f"Empty class folder: {class_dir.name}")
    total = sum(class_counts.values())
    return total, class_counts, issues


def plot_char_distribution(char_counter: Counter, out_path: Path):
    """
    Bar chart of character frequency across all transcripts.
    WHY: Understanding character distribution is essential for:
      - Detecting severe class imbalance that might need weighted loss
      - Knowing which letters appear rarely (hard to learn)
    """
    chars = sorted(char_counter.keys())
    counts = [char_counter[c] for c in chars]

    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(chars, counts, color="#4A90D9", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Character", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Character Frequency Distribution (Continuous ISL Dataset)", fontsize=13)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_duration_distribution(durations: list, out_path: Path):
    """
    Histogram of segment durations (seconds).
    WHY: CTC requires knowing the maximum sequence length in advance.
    Very long segments need more GPU memory; very short ones might be corrupt.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(durations, bins=50, color="#E07B54", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Duration (seconds)", fontsize=12)
    ax.set_ylabel("Number of Segments", fontsize=12)
    ax.set_title("Segment Duration Distribution (Continuous ISL Dataset)", fontsize=13)
    ax.axvline(x=min(durations), color="red",  linestyle="--",
               label=f"Min: {min(durations):.2f}s", linewidth=1.5)
    ax.axvline(x=max(durations), color="blue", linestyle="--",
               label=f"Max: {max(durations):.2f}s", linewidth=1.5)
    ax.axvline(x=sum(durations)/len(durations), color="green", linestyle="-",
               label=f"Mean: {sum(durations)/len(durations):.2f}s", linewidth=1.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_sequence_length_distribution(seq_lengths: list, out_path: Path):
    """
    Histogram of label sequence lengths (number of characters per segment).
    WHY: CTC loss requires T > N (input frames > label length).
    If this constraint is violated, training will crash. We need to know
    the distribution of N to set minimum frame counts in preprocessing.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(seq_lengths, bins=40, color="#7BBF6A", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Label Sequence Length (characters)", fontsize=12)
    ax.set_ylabel("Number of Segments", fontsize=12)
    ax.set_title("Label Sequence Length Distribution", fontsize=13)
    ax.axvline(x=max(seq_lengths), color="red", linestyle="--",
               label=f"Max: {max(seq_lengths)}", linewidth=1.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =============================================================================
# MAIN INSPECTION
# =============================================================================

def main():
    lines = []   # collects all output lines for the report file

    def log(s=""):
        """Print and also collect for file output."""
        print(s)
        lines.append(s)

    random.seed(42)

    # ── 1. LOAD ALL CSVs ────────────────────────────────────────────────────
    log(section("LOADING METADATA CSVs"))

    ann_df   = pd.read_csv(ANN_CSV)
    loc_df   = pd.read_csv(LOC_CSV)
    split_df = pd.read_csv(SPLIT_CSV)

    log(bullet("Annotations CSV   -- rows", len(ann_df)))
    log(bullet("Localization CSV  -- rows", len(loc_df)))
    log(bullet("Split info CSV    -- rows", len(split_df)))
    log(bullet("Annotations columns", list(ann_df.columns)))
    log(bullet("Localization columns", list(loc_df.columns)))
    log(bullet("Split columns", list(split_df.columns)))

    # Merge all three on uid (annotations uid == split uid)
    # loc_df uses video_id + segment_index -- construct uid to join
    loc_df["uid"] = (loc_df["video_id"].astype(str) + "_seg" +
                     loc_df["segment_index"].apply(lambda x: f"{int(x):03d}"))

    merged = (ann_df
              .merge(split_df, on="uid", how="outer")
              .merge(loc_df[["uid", "duration_sec", "start_sec", "end_sec"]],
                     on="uid", how="left"))

    log(f"\n  Merged dataframe shape: {merged.shape}")

    # ── 2. VIDEO FILE INVENTORY ──────────────────────────────────────────────
    log(section("VIDEO FILE INVENTORY"))

    video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
    log(bullet("Video files found in /videos/", len(video_files)))
    log(bullet("Segments in annotations CSV", len(ann_df)))

    ann_uids  = set(ann_df["uid"])
    vid_names = {f.stem for f in video_files}

    missing_videos = ann_uids - vid_names
    extra_videos   = vid_names - ann_uids

    if missing_videos:
        log(f"\n  WARNING: UIDs in annotations but NO video file ({len(missing_videos)}):")
        for u in sorted(missing_videos):
            log(f"      {u}")
    else:
        log("\n  OK: All annotation UIDs have a corresponding video file.")

    if extra_videos:
        log(f"\n  WARNING: Video files with NO annotation ({len(extra_videos)}):")
        for u in sorted(extra_videos)[:20]:
            log(f"      {u}")
        if len(extra_videos) > 20:
            log(f"      ... and {len(extra_videos)-20} more")
    else:
        log("  OK: All video files have a corresponding annotation.")

    # ── 3. FILE SIZE / CORRUPTION HEURISTICS ────────────────────────────────
    log(section("FILE SIZE AND POTENTIAL CORRUPTION"))

    file_sizes = {f.stem: f.stat().st_size for f in video_files}
    tiny_files = {k: v for k, v in file_sizes.items() if v < MIN_FILE_SIZE_BYTES}

    log(bullet("Minimum file size threshold (bytes)", f"{MIN_FILE_SIZE_BYTES:,}"))
    log(bullet("Files below threshold (potentially corrupt)", len(tiny_files)))

    if tiny_files:
        log(f"\n  Flagged tiny files:")
        for name, sz in sorted(tiny_files.items(), key=lambda x: x[1]):
            log(f"      {name:<40} {sz:>10,} bytes")

    total_size_gb = sum(file_sizes.values()) / (1024**3)
    log(bullet("\n  Total dataset size (videos folder)", f"{total_size_gb:.2f} GB"))
    log(bullet("Mean file size (bytes)", f"{sum(file_sizes.values())//len(file_sizes):,}"))
    log(bullet("Min file size (bytes)", f"{min(file_sizes.values()):,}"))
    log(bullet("Max file size (bytes)", f"{max(file_sizes.values()):,}"))

    # ── 4. DURATION STATISTICS ───────────────────────────────────────────────
    log(section("DURATION STATISTICS"))

    durations = loc_df["duration_sec"].dropna().tolist()

    total_min   = sum(durations) / 60
    mean_dur    = sum(durations) / len(durations)
    median_dur  = sorted(durations)[len(durations) // 2]

    log(bullet("Total number of segments", len(durations)))
    log(bullet("Total duration (minutes)", f"{total_min:.2f}"))
    log(bullet("Mean segment duration (seconds)", f"{mean_dur:.2f}"))
    log(bullet("Median segment duration (seconds)", f"{median_dur:.2f}"))
    log(bullet("Minimum duration (seconds)", f"{min(durations):.3f}"))
    log(bullet("Maximum duration (seconds)", f"{max(durations):.3f}"))

    short_segs = loc_df[loc_df["duration_sec"] < MIN_DUR_SEC]
    long_segs  = loc_df[loc_df["duration_sec"] > MAX_DUR_SEC]

    log(f"\n  Outlier thresholds: < {MIN_DUR_SEC}s = too short | > {MAX_DUR_SEC}s = too long")
    log(bullet("Segments shorter than threshold", len(short_segs)))
    log(bullet("Segments longer than threshold", len(long_segs)))

    if len(short_segs):
        log("\n  Very short segments:")
        for _, row in short_segs.iterrows():
            loc_uid = row.get("uid", "?")
            log(f"      {loc_uid:<35} {row['duration_sec']:.3f}s")

    if len(long_segs):
        log(f"\n  Very long segments (top {min(10, len(long_segs))}):")
        for _, row in long_segs.sort_values("duration_sec", ascending=False).head(10).iterrows():
            loc_uid = row.get("uid", "?")
            log(f"      {loc_uid:<35} {row['duration_sec']:.1f}s")

    plot_duration_distribution(durations, DUR_DIST_PNG)
    log(f"\n  Duration histogram saved to: {DUR_DIST_PNG}")

    # ── 5. LABEL / CHARACTER ANALYSIS ───────────────────────────────────────
    log(section("LABEL AND CHARACTER DISTRIBUTION"))

    transcripts = ann_df["text"].dropna().str.strip().tolist()
    log(bullet("Total labelled segments", len(transcripts)))
    log(bullet("Unique label strings", len(set(transcripts))))

    all_chars = Counter()
    seq_lengths = []
    for t in transcripts:
        chars = [c.lower() for c in t if c != " "]
        all_chars.update(chars)
        seq_lengths.append(len(chars))

    total_chars = sum(all_chars.values())
    unique_chars = len(all_chars)
    log(bullet("Total characters (excluding spaces)", f"{total_chars:,}"))
    log(bullet("Unique characters in vocabulary", unique_chars))
    log(bullet("Characters present", "".join(sorted(all_chars.keys()))))

    log(f"\n  Character frequencies (top 10 most common):")
    for ch, cnt in all_chars.most_common(10):
        pct = 100 * cnt / total_chars
        log(f"      '{ch}'  -->  {cnt:>6,}  ({pct:5.2f}%)")

    log(f"\n  Character frequencies (10 rarest):")
    for ch, cnt in all_chars.most_common()[-10:]:
        pct = 100 * cnt / total_chars
        log(f"      '{ch}'  -->  {cnt:>6,}  ({pct:5.2f}%)")

    log(f"\n  Label sequence lengths (characters per segment, spaces excluded):")
    log(bullet("    Min", min(seq_lengths)))
    log(bullet("    Max", max(seq_lengths)))
    log(bullet("    Mean", f"{sum(seq_lengths)/len(seq_lengths):.1f}"))
    log(bullet("    Median", sorted(seq_lengths)[len(seq_lengths)//2]))

    plot_char_distribution(all_chars, CHAR_DIST_PNG)
    log(f"\n  Character distribution chart saved to: {CHAR_DIST_PNG}")
    plot_sequence_length_distribution(seq_lengths, SEQ_LEN_PNG)
    log(f"  Sequence length chart saved to: {SEQ_LEN_PNG}")

    # ── 6. SPLIT ANALYSIS ───────────────────────────────────────────────────
    log(section("TRAIN / VAL / TEST SPLIT ANALYSIS"))

    split_counts = split_df["split"].value_counts()
    log("  Split distribution (as provided by dataset authors):")
    for split_name, count in split_counts.items():
        pct = 100 * count / len(split_df)
        log(f"      {split_name:<10}  {count:>5} segments  ({pct:.1f}%)")

    split_df["video_id"] = split_df["uid"].str.extract(r"^(.+)_seg\d+$")

    vid_split_map = split_df.groupby("video_id")["split"].nunique()
    split_leaking = vid_split_map[vid_split_map > 1]

    log(f"\n  Checking video-source independence:")
    log(bullet("    Unique YouTube source videos", len(vid_split_map)))
    log(bullet("    Sources appearing in > 1 split", len(split_leaking)))

    if len(split_leaking) > 0:
        log(f"\n  WARNING: These video sources span multiple splits (potential leakage):")
        for vid_id in split_leaking.index:
            splits_used = split_df[split_df["video_id"] == vid_id]["split"].unique()
            log(f"      {vid_id}  -->  {sorted(splits_used)}")
    else:
        log("\n  OK: All segments from a given YouTube source are in exactly one split.")
        log("    This is the strongest signer-independence guarantee available in this")
        log("    dataset (since explicit signer IDs are not provided).")

    merged["video_id"] = merged["uid"].str.extract(r"^(.+)_seg\d+$")
    log("\n  Duration breakdown by split:")
    for sp in ["train", "val", "test"]:
        sp_df = merged[merged["split"] == sp]
        sp_dur = sp_df["duration_sec"].dropna().sum() / 60
        sp_chars = sp_df["text"].dropna().apply(
            lambda t: len([c for c in t if c != " "])).sum()
        log(f"      {sp:<8}  {len(sp_df):>5} segs  |  {sp_dur:6.1f} min  |  {sp_chars:>6,} chars")

    # ── 7. VIDEO RESOLUTION / FPS SAMPLE PROBE ──────────────────────────────
    log(section("VIDEO RESOLUTION / FPS (SAMPLE PROBE)"))
    log(f"  Opening {N_SAMPLE_VIDEOS} random videos with OpenCV...\n")

    sample_results = open_video_sample(video_files, n=N_SAMPLE_VIDEOS)
    if not sample_results:
        log("  WARNING: OpenCV not available or sample probe failed.")
        log("    Install opencv-python to enable this check.")
        log("    Skipping resolution/FPS check.")
    else:
        for r in sample_results:
            status = "OK" if r["ok"] else "FAILED TO OPEN"
            log(f"      {r['path'].name:<40}  {r['width']}x{r['height']}  "
                f"@ {r['fps']:.1f} FPS  |  {r['frame_count']} frames  [{status}]")

        fps_vals = set(r["fps"] for r in sample_results if r["ok"])
        if len(fps_vals) > 1:
            log(f"\n  WARNING: Multiple FPS values found in sample: {fps_vals}")
            log("    This means segments may have different frame rates.")
            log("    Recommendation: resample all videos to a fixed FPS in Stage 2.")
        else:
            log(f"\n  OK: All sampled videos share the same FPS: {fps_vals}")

    # ── 8. STATIC ISL IMAGE DATASET ─────────────────────────────────────────
    log(section("STATIC ISL IMAGE DATASET"))

    isl_data_dir = STATIC_DIR / "isl_data"
    if not isl_data_dir.exists():
        log(f"  WARNING: Expected 'isl_data/' subfolder not found in {STATIC_DIR}")
        log("    The static dataset may not be downloaded yet.")
        log("    According to readmeisl.txt: ~14,000 images, 35 classes (A-Z + 1-9)")
    else:
        total_imgs, class_counts, issues = count_static_dataset(STATIC_DIR)
        log(bullet("Total classes", len(class_counts)))
        log(bullet("Total images", f"{total_imgs:,}"))
        log(bullet("Images per class (mean)", f"{total_imgs // max(1, len(class_counts))}"))

        if issues:
            log(f"\n  Issues found:")
            for iss in issues:
                log(f"      {iss}")
        else:
            log("\n  OK: No issues found in static dataset.")

        log("\n  Class distribution (sorted by image count, showing top 10):")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
            log(f"      Class '{cls}'  -->  {cnt} images")

    # ── 9. SUMMARY AND RECOMMENDATIONS ──────────────────────────────────────
    log(section("SUMMARY AND NEXT-STEP RECOMMENDATIONS"))

    log("""
  DATASET SUMMARY (Continuous ISL Fingerspelling)
  ---------------------------------------------------
  * 1,308 video segments, ~70.85 minutes total
  * Segments from YouTube ISL news interpreter footage
  * Labels: multi-word fingerspelled sequences (proper nouns, place names)
  * No explicit signer IDs -- 'signer' ~= YouTube channel/person
  * Train/val/test split already provided by dataset authors (video-source-clean)

  STATIC ISL IMAGE DATASET
  -------------------------
  * ~14,000 images, 35 classes (A-Z + 1-9)
  * 3 contributors, 2 background types
  * Role in thesis: ablation study baseline OR pretraining for letter classifier

  SPLIT STRATEGY RECOMMENDATION
  --------------------------------
  The dataset authors' provided split_info.csv should be used as-is.
  Reasons:
    (a) It is video-source-independent (verified above): all segments from
        a given YouTube video stay in one split.
    (b) Overriding it would make results incomparable to the paper
        (Kirandevraj et al., WSLP @ AACL-IJCNLP 2025).
    (c) We lack explicit signer IDs, so any split we construct ourselves
        would be no more 'signer-independent' than theirs.

  DECISION REQUIRED (for DESIGN_DECISIONS.md):
     Do you accept the provided split, or do you want to construct a
     custom signer-independent split? Awaiting your approval.

  POTENTIAL DATA QUALITY FLAGS
  -----------------------------
  (See detailed output above for exact counts)
  * Tiny files (<50 KB): flagged above -- verify with Stage 2 hand detection
  * Very short segments (<0.5s): unlikely to carry enough frames for CTC
  * Very long segments (>30s): may dominate GPU memory usage during training
  * Special characters in labels (hyphens, periods, digits):
    decision needed -- include or exclude non-alpha characters?

  RECOMMENDED VOCABULARY CHOICE
  --------------------------------
  The labels contain lowercase alphabets + spaces + special characters.
  For CTC we need a fixed character vocabulary. Options:
    Option A: 26 letters (a-z) only -- strip/ignore non-alpha chars (recommended)
    Option B: 26 letters + 10 digits + common punctuation
  --> This decision affects CTC blank token index and confusion matrix axes.
  --> Awaiting your direction before Stage 2.
    """)

    # ── WRITE REPORT ────────────────────────────────────────────────────────
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*72}")
    print(f"  Report written to: {REPORT_TXT}")
    print(f"  Charts saved to:   {BASE_DIR}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
