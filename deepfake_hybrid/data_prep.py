"""Prepare FF++, SDFVD, and the simulated c40-style test.

Run from the project folder (the one that contains the deepfake_hybrid package):

    python -m deepfake_hybrid.data_prep --step inspect
    python -m deepfake_hybrid.data_prep --step ffpp
    python -m deepfake_hybrid.data_prep --step sdfvd
    python -m deepfake_hybrid.data_prep --step c40

    python -m deepfake_hybrid.data_prep --step all

No model is trained here. Counts printed by a run are the real counts;
nothing in this file sets a target accuracy or a fixed video quota.

FF++ frames in the Hugging Face mirror are already extracted. Each frame_idx
is kept. SDFVD is a set of videos, so that step grabs every 5th frame.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

# Point the Hugging Face cache at Drive before the datasets import.
import deepfake_hybrid.config as config

os.environ.setdefault("HF_HOME", config.HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(config.HF_HOME, "datasets"))

import numpy as np
from PIL import Image
from tqdm import tqdm

from deepfake_hybrid.faces import FacePrep, infer_face_mode
from deepfake_hybrid.frequency import frequency_map


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Data prep still runs if torch is imported later by facenet.
        pass


def log(message: str) -> None:
    print(message, flush=True)
    path = os.path.join(config.OUTPUT_DIR, config.DIR_LOGS, "data_prep.log")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def save_json(name: str, payload) -> str:
    folder = os.path.join(config.OUTPUT_DIR, config.DIR_MANIFESTS)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    log(f"saved {path}")
    return path


def load_json(name: str):
    path = os.path.join(config.OUTPUT_DIR, config.DIR_MANIFESTS, name)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def rel_to_output(path: str) -> str:
    return os.path.relpath(path, config.OUTPUT_DIR).replace("\\", "/")


def safe_name(video: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", str(video)).strip("_")
    return cleaned or "video"


def person_ids(video: str) -> list[str]:
    """Numeric ids in a FaceForensics++ name. '000_003' -> ['000', '003']."""
    found = re.findall(r"\d+", str(video))
    return found if found else [str(video)]


def identity_source(video: str) -> str:
    """Drop a 'Category/' prefix so only the FF++ name supplies person ids."""
    text = str(video)
    if "/" in text:
        return text.split("/", 1)[1]
    return text


def source_identity(video: str) -> str:
    return person_ids(identity_source(video))[0]


def output_path(*parts: str) -> str:
    path = os.path.join(config.OUTPUT_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item

    def find(self, item: str) -> str:
        self.add(item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            # Keep the smaller id as the root so group names are stable.
            if root_left <= root_right:
                self.parent[root_right] = root_left
            else:
                self.parent[root_left] = root_right


# ---------------------------------------------------------------------------
# Settings fingerprint, so a resumed run cannot mix two different crops
# ---------------------------------------------------------------------------

def prep_settings() -> dict:
    return {
        "image_size": config.IMAGE_SIZE,
        "mtcnn_margin": config.MTCNN_MARGIN,
        "mtcnn_keep_prob": config.MTCNN_KEEP_PROB,
        "min_frames": config.MIN_FRAMES_PER_VIDEO,
        "seq_len": config.SEQ_LEN,
        "seq_stride": config.SEQ_STRIDE,
        "max_seqs": config.MAX_SEQS_PER_VIDEO,
        "seed": config.SEED,
        "pool_splits": list(config.POOL_SPLITS),
        "identity_grouping": config.IDENTITY_GROUPING,
        "face_input_mode": config.FACE_INPUT_MODE,
        "train_ratio": config.TRAIN_RATIO,
        "val_ratio": config.VAL_RATIO,
        "test_ratio": config.TEST_RATIO,
        "sdfvd_frame_every": config.SDFVD_FRAME_EVERY,
        "sdfvd_max_frames": config.SDFVD_MAX_FRAMES,
        "c40_crf": config.C40_CRF,
        "c40_label": config.C40_LABEL,
    }


def check_settings() -> None:
    path = os.path.join(config.OUTPUT_DIR, config.DIR_MANIFESTS, "prep_settings.json")
    current = prep_settings()
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        previous = json.load(handle)
    if previous != current:
        raise RuntimeError(
            "config.py changed after a previous prep run, and cached faces would "
            "no longer match the new settings. Delete OUTPUT_DIR and run again, "
            f"or restore the old settings. Previous={previous} current={current}"
        )


def write_settings() -> None:
    save_json("prep_settings.json", prep_settings())


# ---------------------------------------------------------------------------
# Balance and identity split (no dataset access; tested on their own)
# ---------------------------------------------------------------------------

def balance_videos(videos_by_category: dict[str, list[str]], seed: int) -> dict[str, list[str]]:
    """Use every eligible real video, and the same number of fake videos.

    Fake videos are spread one-by-one across the four manipulation types.
    If one type runs out, later picks come from the types that still have videos.
    If fakes are scarcer than reals, real videos are sampled down to match.
    """
    rng = random.Random(seed)
    real = list(videos_by_category.get(config.REAL_CATEGORY, []))
    pools = {}
    for category in config.FAKE_CATEGORIES:
        pool = list(videos_by_category.get(category, []))
        rng.shuffle(pool)
        pools[category] = pool

    n_fake_available = sum(len(pool) for pool in pools.values())
    n_target = len(real)
    if n_fake_available < len(real):
        log(
            f"Fake videos ({n_fake_available}) are fewer than real videos "
            f"({len(real)}). Sampling real videos down so the two counts match."
        )
        rng.shuffle(real)
        real = real[:n_fake_available]
        n_target = len(real)

    selected_fake = {category: [] for category in config.FAKE_CATEGORIES}
    for _ in range(n_target):
        available = [category for category in config.FAKE_CATEGORIES if pools[category]]
        if not available:
            break
        # Fewest already chosen, then the name, so the spread stays even and stable.
        available.sort(key=lambda category: (len(selected_fake[category]), category))
        chosen = available[0]
        selected_fake[chosen].append(pools[chosen].pop())

    selected = {config.REAL_CATEGORY: sorted(real)}
    for category in config.FAKE_CATEGORIES:
        selected[category] = sorted(selected_fake[category])
    return selected


def group_videos(video_ids: list[str], grouping: str) -> dict[str, list[str]]:
    """Map a group id to the videos that must share a split."""
    if grouping == "source":
        groups = defaultdict(list)
        for video in video_ids:
            groups[source_identity(video)].append(video)
        return dict(groups)

    if grouping != "all_persons":
        raise ValueError(f"unknown IDENTITY_GROUPING: {grouping}")

    # Union every id that co-occurs in a filename, then union videos that share an id.
    people = UnionFind()
    for video in video_ids:
        ids = person_ids(identity_source(video))
        for other in ids[1:]:
            people.union(ids[0], other)
    groups = defaultdict(list)
    for video in video_ids:
        root = people.find(person_ids(identity_source(video))[0])
        groups[root].append(video)
    return dict(groups)


def assign_groups(groups: dict[str, list[str]], seed: int) -> dict[str, list[str]]:
    """Place whole identity groups into train / val / test, near 70/15/15."""
    ratio_sum = config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {ratio_sum}")

    items = list(groups.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)

    n_videos = sum(len(videos) for _, videos in items)
    targets = {
        "train": config.TRAIN_RATIO * n_videos,
        "val": config.VAL_RATIO * n_videos,
        "test": config.TEST_RATIO * n_videos,
    }
    counts = {name: 0 for name in targets}
    assigned = {name: [] for name in targets}
    for _group_id, videos in items:
        split_name = max(targets, key=lambda name: targets[name] - counts[name])
        assigned[split_name].extend(videos)
        counts[split_name] += len(videos)

    if any(len(assigned[name]) == 0 for name in ("train", "val", "test")):
        sizes = sorted((len(videos) for videos in groups.values()), reverse=True)
        raise RuntimeError(
            "Identity grouping left a split empty, so a 70/15/15 split with "
            "zero person overlap is not possible on this subset. Largest group "
            f"sizes: {sizes[:8]}. Set IDENTITY_GROUPING = 'source' in config.py "
            "and delete the manifests before re-running."
        )
    return assigned


def assert_split_disjoint(split_of: dict[str, str], grouping: str) -> dict:
    """Fail if a video or a protected identity id lands in two splits."""
    videos = defaultdict(set)
    protected = defaultdict(set)
    leaked_targets = defaultdict(set)
    for video, split_name in split_of.items():
        videos[split_name].add(video)
        ids = person_ids(identity_source(video))
        if grouping == "source":
            protected[split_name].add(ids[0])
            leaked_targets[split_name].update(ids[1:])
        else:
            protected[split_name].update(ids)

    names = ("train", "val", "test")
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            video_overlap = videos[left] & videos[right]
            id_overlap = protected[left] & protected[right]
            if video_overlap or id_overlap:
                raise AssertionError(
                    f"overlap between {left} and {right}: "
                    f"videos={sorted(video_overlap)[:10]} ids={sorted(id_overlap)[:10]}"
                )

    target_leak = []
    if grouping == "source":
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                target_leak.extend(sorted(leaked_targets[left] & leaked_targets[right]))
                # A target id in one split may be a source id in another.
                target_leak.extend(sorted(leaked_targets[left] & protected[right]))
                target_leak.extend(sorted(protected[left] & leaked_targets[right]))
    return {
        "grouping": grouping,
        "videos_per_split": {name: len(videos[name]) for name in names},
        "identities_per_split": {name: len(protected[name]) for name in names},
        "target_ids_shared_across_splits": sorted(set(target_leak)),
    }


def sequence_windows(frames: list[dict]) -> list[list[dict]]:
    """Non-overlapping windows of SEQ_LEN on frames sorted by frame_idx.

    Dropped (no-face) frames are already absent. A window is the next
    SEQ_LEN kept frames in time order, then the index jumps by SEQ_STRIDE.
    """
    ordered = sorted(frames, key=lambda frame: frame["frame_idx"])
    windows = []
    start = 0
    while start + config.SEQ_LEN <= len(ordered) and len(windows) < config.MAX_SEQS_PER_VIDEO:
        windows.append(ordered[start : start + config.SEQ_LEN])
        start += config.SEQ_STRIDE
    return windows


# ---------------------------------------------------------------------------
# Hugging Face mirror
# ---------------------------------------------------------------------------

def _as_category(feature, value) -> str:
    if hasattr(feature, "int2str") and isinstance(value, (int, np.integer)):
        return str(feature.int2str(int(value)))
    return str(value)


def load_pooled_dataset():
    from datasets import Image as HFImage
    from datasets import load_dataset

    log(f"Loading {config.HF_DATASET_ID}")
    dataset = load_dataset(config.HF_DATASET_ID)
    log(f"splits: { {name: len(dataset[name]) for name in dataset.keys()} }")
    log(f"features: {dataset[next(iter(dataset.keys()))].features}")

    missing = [name for name in config.POOL_SPLITS if name not in dataset]
    if missing:
        raise RuntimeError(f"POOL_SPLITS {missing} are not in the dataset. Found {list(dataset.keys())}")

    skipped = [name for name in dataset.keys() if name not in config.POOL_SPLITS]
    if skipped:
        skipped_sizes = {name: len(dataset[name]) for name in skipped}
        log(
            f"Leaving these official splits out of the pool: {skipped_sizes}. "
            "Edit POOL_SPLITS in config.py to include them."
        )

    pooled = {}
    for name in config.POOL_SPLITS:
        pooled[name] = dataset[name].cast_column("image", HFImage(decode=False))
    return pooled


def _column_lists(part, column_names: list[str]) -> dict:
    return part.select_columns(column_names).to_dict()


def decode_hf_image(cell) -> Image.Image:
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        if cell.get("path"):
            return Image.open(cell["path"]).convert("RGB")
    raise TypeError(f"cannot decode image cell of type {type(cell)}")


def collect_metadata(pooled) -> tuple[list[dict], dict]:
    """One row per mirror frame in the pooled splits, plus category counts."""
    rows = []
    counts = {}
    for split_name, part in pooled.items():
        columns = _column_lists(part, ["label", "category", "video", "frame_idx"])
        category_feature = part.features["category"]
        label_feature = part.features["label"]
        n = len(columns["video"])
        split_counter = Counter()
        for index in range(n):
            category = _as_category(category_feature, columns["category"][index])
            label_text = _as_category(label_feature, columns["label"][index])
            video = str(columns["video"][index])
            frame_idx = int(columns["frame_idx"][index])
            split_counter[category] += 1
            rows.append(
                {
                    "hf_split": split_name,
                    "row_index": index,
                    "video": video,
                    "frame_idx": frame_idx,
                    "category": category,
                    "label_text": label_text,
                }
            )
        counts[split_name] = dict(split_counter)
        log(f"category counts in HF split {split_name}: {dict(split_counter)}")
    return rows, counts


def step_inspect(pooled=None, rows=None, counts=None) -> tuple[str, list[dict]]:
    """Print features, splits, and category counts. Save one sample frame."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    log("=== inspect ===")
    if pooled is None:
        pooled = load_pooled_dataset()
    if rows is None or counts is None:
        rows, counts = collect_metadata(pooled)
    save_json("ffpp_category_counts.json", counts)

    first_split = config.POOL_SPLITS[0]
    sample_cell = pooled[first_split][0]["image"]
    sample = decode_hf_image(sample_cell)
    sample_path = output_path(config.DIR_INSPECT, "sample_frame.jpg")
    sample.save(sample_path, quality=95)
    auto_mode = infer_face_mode(sample.width, sample.height)
    mode = auto_mode if config.FACE_INPUT_MODE == "auto" else config.FACE_INPUT_MODE
    info = {
        "path": rel_to_output(sample_path),
        "width": sample.width,
        "height": sample.height,
        "auto_mode": auto_mode,
        "face_mode_used": mode,
        "hf_split": first_split,
        "row_index": 0,
    }
    save_json("face_mode.json", info)
    log(
        f"sample image {sample.width}x{sample.height} saved to {sample_path}. "
        f"auto_mode={auto_mode}. face step will use {mode}."
    )
    log("Look at the sample. If auto_mode is wrong, set FACE_INPUT_MODE in config.py.")
    return mode, rows


def videos_from_rows(rows: list[dict]) -> dict[str, dict]:
    """Group kept categories by video. Drop videos with fewer than 10 frames."""
    grouped = defaultdict(list)
    label_mismatch = 0
    dropped_rows = Counter()
    raw_categories = defaultdict(set)
    for row in rows:
        if row["category"] in config.DROP_CATEGORIES or row["category"] not in config.KEEP_CATEGORIES:
            dropped_rows[row["category"]] += 1
            continue
        expected = "real" if row["category"] == config.REAL_CATEGORY else "deepfake"
        text = row["label_text"].strip().lower()
        if text not in (expected, "0", "1"):
            # Accept the category as the source of truth, and count disagreements.
            if text not in ("real", "deepfake"):
                label_mismatch += 1
            elif (text == "real") != (expected == "real"):
                label_mismatch += 1
        # The same numeric name (000_003) is reused by every manipulation method.
        # Keying by category keeps those methods as separate videos.
        raw_name = row["video"]
        if raw_name.startswith(row["category"] + "/") or raw_name.startswith(row["category"] + "_"):
            video_key = raw_name
        else:
            video_key = f"{row['category']}/{raw_name}"
        row["source_video"] = raw_name
        row["video_key"] = video_key
        raw_categories[raw_name].add(row["category"])
        grouped[video_key].append(row)

    log(f"rows dropped by category filter: {dict(dropped_rows)}")
    if label_mismatch:
        log(f"rows whose label text disagrees with category: {label_mismatch}")
    shared = sum(1 for cats in raw_categories.values() if len(cats) > 1)
    log(f"raw video ids that appear in more than one kept category: {shared}")

    videos = {}
    short = 0
    for video, frames in grouped.items():
        # One mirror can list the same frame_idx twice across pooled splits.
        unique = {}
        for frame in frames:
            unique[(frame["hf_split"], frame["frame_idx"])] = frame
        ordered = sorted(unique.values(), key=lambda frame: (frame["frame_idx"], frame["hf_split"]))
        # Prefer a single copy of each frame_idx.
        by_idx = {}
        for frame in ordered:
            by_idx.setdefault(frame["frame_idx"], frame)
        ordered = [by_idx[idx] for idx in sorted(by_idx)]
        if len(ordered) < config.MIN_FRAMES_PER_VIDEO:
            short += 1
            continue
        category = ordered[0]["category"]
        source_video = ordered[0]["source_video"]
        videos[video] = {
            "video": video,
            "source_video": source_video,
            "category": category,
            "label": config.LABEL_REAL if category == config.REAL_CATEGORY else config.LABEL_FAKE,
            # Identity comes from the numeric FF++ name, not the category prefix.
            "source_identity": source_identity(source_video),
            "person_ids": person_ids(source_video),
            "n_frames": len(ordered),
            "frames": ordered,
        }
    log(f"videos dropped for having < {config.MIN_FRAMES_PER_VIDEO} frames: {short}")
    return videos


def step_ffpp() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    check_settings()
    log("=== ffpp ===")
    set_seed(config.SEED)
    pooled = load_pooled_dataset()
    rows, counts = collect_metadata(pooled)
    mode, rows = step_inspect(pooled, rows, counts)
    videos = videos_from_rows(rows)

    by_category = defaultdict(list)
    for video, record in videos.items():
        by_category[record["category"]].append(video)
    before = {category: len(ids) for category, ids in sorted(by_category.items())}
    log(f"videos with >= {config.MIN_FRAMES_PER_VIDEO} frames, before balance: {before}")

    selected = balance_videos(by_category, config.SEED)
    balance_counts = {
        category: len(ids) for category, ids in selected.items()
    }
    balance_counts["real_videos"] = balance_counts.get(config.REAL_CATEGORY, 0)
    balance_counts["fake_videos"] = sum(
        balance_counts.get(category, 0) for category in config.FAKE_CATEGORIES
    )
    log(f"balanced video counts: {balance_counts}")
    save_json("ffpp_balance_counts.json", {"before_balance": before, "selected": balance_counts, "video_ids": selected})

    chosen_ids = [video for ids in selected.values() for video in ids]
    groups = group_videos(chosen_ids, config.IDENTITY_GROUPING)
    group_sizes = sorted((len(members) for members in groups.values()), reverse=True)
    log(
        f"identity grouping={config.IDENTITY_GROUPING} groups={len(groups)} "
        f"largest={group_sizes[:6]}"
    )
    assigned = assign_groups(groups, config.SEED)
    split_of = {}
    for split_name, members in assigned.items():
        for video in members:
            split_of[video] = split_name

    overlap_report = assert_split_disjoint(split_of, config.IDENTITY_GROUPING)
    log(f"split overlap check passed: {overlap_report}")
    video_rows = []
    for video, split_name in sorted(split_of.items()):
        record = videos[video]
        record["split"] = split_name
        video_rows.append(
            {
                "video": video,
                "source_video": record["source_video"],
                "safe_name": safe_name(video),
                "category": record["category"],
                "label": record["label"],
                "split": split_name,
                "source_identity": record["source_identity"],
                "person_ids": record["person_ids"],
                "n_mirror_frames": record["n_frames"],
            }
        )
    save_json("ffpp_videos.json", video_rows)
    if overlap_report["target_ids_shared_across_splits"]:
        log(
            "Source-only grouping shares these target ids across splits: "
            f"{overlap_report['target_ids_shared_across_splits'][:30]}"
        )

    cache_faces(pooled, videos, split_of, mode)
    write_sequence_manifests(videos, split_of, prefix="ffpp")
    write_settings()


def cache_faces(pooled, videos: dict, split_of: dict[str, str], mode: str) -> None:
    """Decode selected frames, save test-split raw PNGs, cache face + FFT."""
    prep = FacePrep(mode)
    log(f"face step: {mode}. MTCNN device will be chosen on the first full-frame batch.")

    # Row lookups for the selected frames only.
    wanted = defaultdict(list)  # hf_split -> list of (row_index, video, frame_idx)
    for video, split_name in split_of.items():
        for frame in videos[video]["frames"]:
            wanted[frame["hf_split"]].append((frame["row_index"], video, frame["frame_idx"]))

    saved = defaultdict(list)
    dropped_no_face = 0
    resumed = 0
    batch_size = 8

    for hf_split, requests in wanted.items():
        part = pooled[hf_split]
        requests.sort(key=lambda item: item[0])
        pending = []

        def flush():
            nonlocal dropped_no_face, resumed
            if not pending:
                return
            images = [item["image"] for item in pending]
            faces = prep.prepare_many(images)
            for item, face in zip(pending, faces):
                if face is None:
                    dropped_no_face += 1
                    continue
                _write_face_and_freq(item, face, saved)
            pending.clear()

        for row_index, video, frame_idx in tqdm(requests, desc=f"ffpp {hf_split}"):
            record = videos[video]
            face_path, freq_path = _face_paths(record["split"], video, frame_idx)
            raw_path = _raw_test_path(video, frame_idx) if record["split"] == "test" else None
            face_ready = os.path.exists(face_path) and os.path.exists(freq_path)
            raw_ready = raw_path is None or os.path.exists(raw_path)
            if face_ready and raw_ready:
                saved[video].append(
                    {"frame_idx": frame_idx, "face_path": rel_to_output(face_path), "freq_path": rel_to_output(freq_path)}
                )
                resumed += 1
                continue

            image = decode_hf_image(part[row_index]["image"])
            if raw_path is not None and not os.path.exists(raw_path):
                image.save(raw_path, format="PNG")
            if face_ready:
                saved[video].append(
                    {"frame_idx": frame_idx, "face_path": rel_to_output(face_path), "freq_path": rel_to_output(freq_path)}
                )
                continue
            pending.append(
                {
                    "image": image,
                    "video": video,
                    "frame_idx": frame_idx,
                    "split": record["split"],
                    "face_path": face_path,
                    "freq_path": freq_path,
                }
            )
            if len(pending) >= batch_size:
                flush()
        flush()

    for video, frames in saved.items():
        unique = {}
        for frame in frames:
            unique[frame["frame_idx"]] = frame
        videos[video]["kept_frames"] = [unique[idx] for idx in sorted(unique)]
    log(f"resumed existing face files: {resumed}")
    log(f"frames dropped because no face passed the threshold: {dropped_no_face}")
    if mode == "full_frame":
        log(f"MTCNN ran on device={getattr(prep, 'device', 'not loaded')}")


def _face_paths(split_name: str, video: str, frame_idx: int) -> tuple[str, str]:
    stem = f"{int(frame_idx):06d}"
    folder = safe_name(video)
    face = output_path(config.DIR_FFPP_FACES, split_name, folder, stem + ".jpg")
    freq = output_path(config.DIR_FFPP_FREQ, split_name, folder, stem + ".npy")
    return face, freq


def _raw_test_path(video: str, frame_idx: int) -> str:
    return output_path(config.DIR_FFPP_RAW_TEST, safe_name(video), f"{int(frame_idx):06d}.png")


def _write_face_and_freq(item: dict, face: np.ndarray, saved: dict) -> None:
    Image.fromarray(face).save(item["face_path"], quality=95)
    np.save(item["freq_path"], frequency_map(face))
    saved[item["video"]].append(
        {
            "frame_idx": item["frame_idx"],
            "face_path": rel_to_output(item["face_path"]),
            "freq_path": rel_to_output(item["freq_path"]),
        }
    )


def write_sequence_manifests(videos: dict, split_of: dict[str, str], prefix: str) -> dict:
    manifests = {name: [] for name in ("train", "val", "test")}
    summary = {
        name: {"videos": 0, "videos_with_sequences": 0, "sequences": 0, "real_sequences": 0, "fake_sequences": 0}
        for name in manifests
    }
    for video, split_name in split_of.items():
        record = videos[video]
        summary[split_name]["videos"] += 1
        frames = record.get("kept_frames", [])
        windows = sequence_windows(frames)
        if not windows:
            continue
        summary[split_name]["videos_with_sequences"] += 1
        for number, window in enumerate(windows):
            manifests[split_name].append(
                {
                    "video": video,
                    "source_video": record.get("source_video", video),
                    "safe_name": safe_name(video),
                    "category": record["category"],
                    "label": record["label"],
                    "split": split_name,
                    "source_identity": record["source_identity"],
                    "person_ids": record["person_ids"],
                    "seq_index": number,
                    "frames": window,
                }
            )
            summary[split_name]["sequences"] += 1
            if record["label"] == config.LABEL_REAL:
                summary[split_name]["real_sequences"] += 1
            else:
                summary[split_name]["fake_sequences"] += 1

    for split_name, rows in manifests.items():
        save_json(f"{prefix}_sequences_{split_name}.json", rows)
    save_json(f"{prefix}_split_counts.json", summary)
    log(f"{prefix} sequence counts: {summary}")
    return summary


# ---------------------------------------------------------------------------
# SDFVD
# ---------------------------------------------------------------------------

def _print_tree(root: str, limit: int = 200) -> str:
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        depth = os.path.relpath(dirpath, root).count(os.sep)
        if os.path.relpath(dirpath, root) == ".":
            depth = 0
        if depth > 3:
            continue
        indent = "  " * depth
        video_files = [name for name in filenames if name.lower().endswith(config.SDFVD_VIDEO_EXTS)]
        other = len(filenames) - len(video_files)
        lines.append(f"{indent}{os.path.basename(dirpath) or root}/  videos={len(video_files)} other_files={other}")
        for name in sorted(video_files)[:5]:
            lines.append(f"{indent}  {name}")
        if len(video_files) > 5:
            lines.append(f"{indent}  ... {len(video_files) - 5} more videos")
        if len(lines) >= limit:
            lines.append("... tree truncated")
            break
    text = "\n".join(lines)
    log("SDFVD folder tree:\n" + text)
    tree_path = output_path(config.DIR_INSPECT, "sdfvd_tree.txt")
    with open(tree_path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    return text


def _label_from_path(path: str, unzip_root: str) -> int | None:
    rel_parts = [part.lower() for part in Path_parts(os.path.relpath(path, unzip_root))]
    folder_parts = rel_parts[:-1]
    real_hit = any(part in config.SDFVD_REAL_DIR_NAMES for part in folder_parts)
    fake_hit = any(part in config.SDFVD_FAKE_DIR_NAMES for part in folder_parts)
    if real_hit and fake_hit:
        return None
    if real_hit:
        return config.LABEL_REAL
    if fake_hit:
        return config.LABEL_FAKE

    # Documented bitmind layout uses v1.mp4 (real) and vs1.mp4 (fake).
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    if re.fullmatch(r"vs\d+", stem):
        return config.LABEL_FAKE
    if re.fullmatch(r"v\d+", stem):
        return config.LABEL_REAL
    return None


def Path_parts(rel: str) -> list[str]:
    parts = []
    while True:
        rel, tail = os.path.split(rel)
        if tail:
            parts.append(tail)
        if not rel or rel == os.sep:
            break
    parts.reverse()
    return parts


def discover_sdfvd_videos(unzip_root: str) -> list[dict]:
    found = []
    unknown = []
    for dirpath, _dirnames, filenames in os.walk(unzip_root):
        for name in sorted(filenames):
            if not name.lower().endswith(config.SDFVD_VIDEO_EXTS):
                continue
            path = os.path.join(dirpath, name)
            label = _label_from_path(path, unzip_root)
            record = {"path": path, "file": name, "label": label}
            if label is None:
                unknown.append(path)
            else:
                found.append(record)
    if unknown:
        preview = "\n".join(unknown[:20])
        raise RuntimeError(
            "Could not tell real from fake for some SDFVD videos. "
            "Set SDFVD_REAL_DIR_NAMES / SDFVD_FAKE_DIR_NAMES in config.py.\n"
            + preview
        )
    return found


def step_sdfvd() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    check_settings()
    log("=== sdfvd ===")
    if not os.path.exists(config.SDFVD_ZIP):
        raise FileNotFoundError(
            f"SDFVD zip not found at {config.SDFVD_ZIP}. "
            "Set DRIVE_ROOT in config.py to the Drive folder that contains sdfvd.zip."
        )
    unzip_root = os.path.join(config.OUTPUT_DIR, config.DIR_SDFVD_UNZIP)
    marker = os.path.join(unzip_root, ".unzipped")
    os.makedirs(unzip_root, exist_ok=True)
    if os.path.exists(marker):
        log(f"SDFVD already unzipped at {unzip_root}")
    else:
        import zipfile

        log(f"Unzipping {config.SDFVD_ZIP} -> {unzip_root}")
        with zipfile.ZipFile(config.SDFVD_ZIP) as archive:
            archive.extractall(unzip_root)
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write(datetime.now(timezone.utc).isoformat() + "\n")
    _print_tree(unzip_root)
    videos = discover_sdfvd_videos(unzip_root)
    n_real = sum(1 for video in videos if video["label"] == config.LABEL_REAL)
    n_fake = sum(1 for video in videos if video["label"] == config.LABEL_FAKE)
    log(f"SDFVD videos found: total={len(videos)} real={n_real} fake={n_fake}")
    log("All of these videos are test-only. No training step reads them.")

    prep = FacePrep("full_frame")
    sequence_rows = []
    per_video = []
    for video in tqdm(videos, desc="sdfvd"):
        sampled = sample_video_frames(video["path"], config.SDFVD_FRAME_EVERY, config.SDFVD_MAX_FRAMES)
        kept = []
        stem = safe_name(os.path.splitext(video["file"])[0])
        label_name = "real" if video["label"] == config.LABEL_REAL else "fake"
        for frame_idx, frame_bgr in sampled:
            face_path = output_path(config.DIR_SDFVD_FACES, label_name, stem, f"{frame_idx:06d}.jpg")
            freq_path = output_path(config.DIR_SDFVD_FREQ, label_name, stem, f"{frame_idx:06d}.npy")
            if os.path.exists(face_path) and os.path.exists(freq_path):
                kept.append(
                    {"frame_idx": frame_idx, "face_path": rel_to_output(face_path), "freq_path": rel_to_output(freq_path)}
                )
                continue
            rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])  # OpenCV is BGR
            image = Image.fromarray(rgb)
            face = prep.prepare(image)
            if face is None:
                continue
            Image.fromarray(face).save(face_path, quality=95)
            np.save(freq_path, frequency_map(face))
            kept.append(
                {"frame_idx": frame_idx, "face_path": rel_to_output(face_path), "freq_path": rel_to_output(freq_path)}
            )
        windows = sequence_windows(kept)
        per_video.append(
            {
                "file": video["file"],
                "label": video["label"],
                "sampled_frames": len(sampled),
                "faces": len(kept),
                "sequences": len(windows),
            }
        )
        for number, window in enumerate(windows):
            sequence_rows.append(
                {
                    "video": video["file"],
                    "safe_name": stem,
                    "category": "sdfvd_real" if video["label"] == config.LABEL_REAL else "sdfvd_fake",
                    "label": video["label"],
                    "split": "test",
                    "dataset": "sdfvd",
                    "seq_index": number,
                    "frames": window,
                }
            )

    n_seq_real = sum(1 for row in sequence_rows if row["label"] == config.LABEL_REAL)
    n_seq_fake = sum(1 for row in sequence_rows if row["label"] == config.LABEL_FAKE)
    summary = {
        "videos": len(videos),
        "real_videos": n_real,
        "fake_videos": n_fake,
        "sequences": len(sequence_rows),
        "real_sequences": n_seq_real,
        "fake_sequences": n_seq_fake,
        "videos_with_no_sequence": sum(1 for row in per_video if row["sequences"] == 0),
    }
    save_json("sdfvd_counts.json", {"summary": summary, "per_video": per_video})
    save_json("sdfvd_sequences_test.json", sequence_rows)
    log(f"SDFVD sequence counts: {summary}")
    write_settings()


def sample_video_frames(path: str, every: int, cap: int) -> list[tuple[int, np.ndarray]]:
    """Read every Nth frame with OpenCV, stop after `cap` sampled frames."""
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    sampled = []
    index = 0
    while len(sampled) < cap:
        ok, frame = capture.read()
        if not ok:
            break
        if index % every == 0:
            sampled.append((index, frame))
        index += 1
    capture.release()
    return sampled


# ---------------------------------------------------------------------------
# c40-style (simulated)
# ---------------------------------------------------------------------------

def step_c40() -> None:
    """Re-encode FF++ test frames with libx264 CRF 40 and crop the decoded frames.

    The source frames are the pre-crop PNGs written by the ffpp step.
    Outputs are labeled c40-style (simulated) everywhere.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    check_settings()
    log(f"=== {config.C40_LABEL} ===")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is not on PATH. Install ffmpeg, or pip install imageio-ffmpeg."
        )

    video_manifest = os.path.join(config.OUTPUT_DIR, config.DIR_MANIFESTS, "ffpp_videos.json")
    if not os.path.exists(video_manifest):
        raise RuntimeError("Run --step ffpp before --step c40. Missing manifests/ffpp_videos.json")
    selected_videos = load_json("ffpp_videos.json")
    test_videos = [row for row in selected_videos if row["split"] == "test"]
    if not test_videos:
        raise RuntimeError("No FF++ test videos in ffpp_videos.json. Run --step ffpp first.")
    face_mode_info = load_json("face_mode.json")
    # Same face step as the c23 subset. Full frames go through the MTCNN box crop.
    mode = face_mode_info["face_mode_used"]
    prep = FacePrep(mode)
    log(f"{config.C40_LABEL} uses face mode '{mode}' (same decision as the FF++ subset).")

    sequence_rows = []
    per_video = []
    for example in tqdm(test_videos, desc=config.C40_LABEL):
        video = example["video"]
        raw_dir = os.path.join(config.OUTPUT_DIR, config.DIR_FFPP_RAW_TEST, safe_name(video))
        if not os.path.isdir(raw_dir):
            raise RuntimeError(f"Missing pre-crop frames for test video {video} at {raw_dir}")
        raw_frames = sorted(
            name for name in os.listdir(raw_dir) if name.lower().endswith(".png")
        )
        decoded_dir = os.path.join(config.OUTPUT_DIR, config.DIR_C40_FRAMES, safe_name(video))
        mp4_path = os.path.join(config.OUTPUT_DIR, config.DIR_C40_VIDEOS, safe_name(video) + ".mp4")
        os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
        os.makedirs(decoded_dir, exist_ok=True)
        encode_and_decode(ffmpeg, raw_dir, raw_frames, mp4_path, decoded_dir)

        decoded_names = sorted(name for name in os.listdir(decoded_dir) if name.lower().endswith(".png"))
        if len(decoded_names) != len(raw_frames):
            log(
                f"{config.C40_LABEL}: {video} wrote {len(raw_frames)} frames and "
                f"read back {len(decoded_names)}. Pairing the shared prefix."
            )
        pair_count = min(len(raw_frames), len(decoded_names))
        kept = []
        for raw_name, decoded_name in zip(raw_frames[:pair_count], decoded_names[:pair_count]):
            frame_idx = int(os.path.splitext(raw_name)[0])
            face_path = output_path(config.DIR_C40_FACES, safe_name(video), f"{frame_idx:06d}.jpg")
            freq_path = output_path(config.DIR_C40_FREQ, safe_name(video), f"{frame_idx:06d}.npy")
            if not (os.path.exists(face_path) and os.path.exists(freq_path)):
                image = Image.open(os.path.join(decoded_dir, decoded_name)).convert("RGB")
                face = prep.prepare(image)
                if face is None:
                    continue
                Image.fromarray(face).save(face_path, quality=95)
                np.save(freq_path, frequency_map(face))
            kept.append(
                {
                    "frame_idx": frame_idx,
                    "face_path": rel_to_output(face_path),
                    "freq_path": rel_to_output(freq_path),
                }
            )
        windows = sequence_windows(kept)
        per_video.append(
            {"video": video, "raw_frames": len(raw_frames), "faces": len(kept), "sequences": len(windows)}
        )
        for number, window in enumerate(windows):
            sequence_rows.append(
                {
                    "video": video,
                    "safe_name": safe_name(video),
                    "category": example["category"],
                    "label": example["label"],
                    "split": "test",
                    "dataset": config.C40_LABEL,
                    "source_identity": example.get("source_identity"),
                    "person_ids": example.get("person_ids"),
                    "seq_index": number,
                    "frames": window,
                }
            )

    n_real = sum(1 for row in sequence_rows if row["label"] == config.LABEL_REAL)
    n_fake = sum(1 for row in sequence_rows if row["label"] == config.LABEL_FAKE)
    summary = {
        "dataset": config.C40_LABEL,
        "crf": config.C40_CRF,
        "codec": "libx264",
        "videos": len(test_videos),
        "sequences": len(sequence_rows),
        "real_sequences": n_real,
        "fake_sequences": n_fake,
        "videos_with_no_sequence": sum(1 for row in per_video if row["sequences"] == 0),
    }
    save_json("c40_style_counts.json", {"summary": summary, "per_video": per_video})
    save_json("c40_style_sequences_test.json", sequence_rows)
    log(f"{config.C40_LABEL} sequence counts: {summary}")
    write_settings()


def encode_and_decode(ffmpeg: str, raw_dir: str, raw_frames: list[str], mp4_path: str, decoded_dir: str) -> None:
    """Write a CRF-40 mp4 from the raw PNGs, then extract the frames ffmpeg reads back.

    frame_idx values are not a contiguous 0..N range, so the encoder gets a
    temporary sequential folder. Decode order follows that same order, and the
    caller pairs decoded frames back to the original frame_idx list.
    """
    decoded_now = sorted(name for name in os.listdir(decoded_dir) if name.lower().endswith(".png"))
    if os.path.exists(mp4_path) and len(decoded_now) == len(raw_frames):
        log(f"reusing {mp4_path}")
        return

    work = os.path.join(config.OUTPUT_DIR, "tmp_c40", safe_name(os.path.basename(mp4_path)))
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work, exist_ok=True)
    try:
        for number, name in enumerate(raw_frames):
            src = os.path.join(raw_dir, name)
            dst = os.path.join(work, f"frame_{number:06d}.png")
            shutil.copyfile(src, dst)
        subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(config.C40_FPS),
                "-start_number", "0",
                "-i", os.path.join(work, "frame_%06d.png"),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264",
                "-crf", str(config.C40_CRF),
                "-pix_fmt", "yuv420p",
                mp4_path,
            ],
            check=True,
        )
        # Clear a previous decode so a short read cannot leave stale frames behind.
        for name in os.listdir(decoded_dir):
            if name.lower().endswith(".png"):
                os.remove(os.path.join(decoded_dir, name))
        subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", mp4_path,
                os.path.join(decoded_dir, "decoded_%06d.png"),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 2 data prep (no training).")
    parser.add_argument(
        "--step",
        default="all",
        choices=["inspect", "ffpp", "sdfvd", "c40", "all"],
        help="Which preparation step to run.",
    )
    args = parser.parse_args(argv)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    log(f"DRIVE_ROOT={config.DRIVE_ROOT}")
    log(f"OUTPUT_DIR={config.OUTPUT_DIR}")
    log(f"SEED={config.SEED}")
    if args.step in ("inspect",):
        step_inspect()
    elif args.step == "ffpp":
        step_ffpp()
    elif args.step == "sdfvd":
        step_sdfvd()
    elif args.step == "c40":
        step_c40()
    else:
        step_ffpp()
        step_sdfvd()
        step_c40()
    log("data prep step finished")


if __name__ == "__main__":
    main(sys.argv[1:])
