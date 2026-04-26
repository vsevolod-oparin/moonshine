"""Download and preprocess Russian speech datasets.

Downloads CV21 ru and RuLS from HuggingFace parquet files.
Extracts audio, resamples to 16kHz mono, normalizes text, writes WAV files and manifests.

Usage:
    python scripts/download_data.py --dataset cv21 --output-dir data/processed
    python scripts/download_data.py --dataset ruls --output-dir data/processed
    python scripts/download_data.py --dataset all --output-dir data/processed
"""

import argparse
import json
import os
import re
import sys
import hashlib
import warnings
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

if not HAS_SOUNDFILE:
    try:
        import scipy.io.wavfile as wavfile
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False


def resample_audio(audio_np, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio_np
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(target_sr, orig_sr)
    up = target_sr // g
    down = orig_sr // g
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    return resample_poly(audio_np, up, down).astype(np.float32)


def save_wav(path, audio, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if HAS_SOUNDFILE:
        sf.write(path, audio, sr)
    else:
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        wavfile.write(path, sr, audio_int16)


def normalize_text(text, abbreviations=None):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[«»""„"␤]', '', text)
    text = re.sub(r'[\(\)\[\]{}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if abbreviations:
        words = text.split()
        expanded = []
        for w in words:
            clean = w.strip('.,;:!?')
            if clean in abbreviations:
                w = w.replace(clean, abbreviations[clean])
            expanded.append(w)
        text = ' '.join(expanded)
    text = re.sub(r'[^\w\sа-яёА-ЯЁ\-.,;:!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_cv21(output_dir, abbreviations):
    dataset_id = "artyomboyko/common_voice_21_0_ru"
    proc_dir = Path(output_dir) / "cv21"
    proc_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "validated": [f"data/validated-{i:05d}-of-00010.parquet" for i in range(10)],
        "train": [f"data/train-{i:05d}-of-00003.parquet" for i in range(3)],
        "test": ["data/test-00000-of-00001.parquet"],
        "validation": ["data/validation-00000-of-00001.parquet"],
    }

    all_records = []
    total_duration = 0.0
    idx = 0

    for split_name, file_list in split_files.items():
        print(f"\n  Processing {split_name} split...")
        for pf in file_list:
            local = hf_hub_download(dataset_id, pf, repo_type="dataset")
            table = pq.read_table(local)
            rows = table.to_pydict()

            for i in range(table.num_rows):
                sentence = rows["sentence"][i]
                if not sentence or not sentence.strip():
                    continue

                audio_data = rows["audio"][i]
                if audio_data is None:
                    continue

                audio_bytes = audio_data.get("bytes") or audio_data.get("path")
                if audio_bytes is None:
                    continue

                client_id = rows["client_id"][i]
                duration_ms = rows["duration[ms]"][i]
                duration_s = (duration_ms or 0) / 1000.0

                if duration_s < 1.0 or duration_s > 30.0:
                    continue

                text = normalize_text(sentence, abbreviations)
                if not text:
                    continue

                try:
                    if HAS_SOUNDFILE:
                        import io
                        audio_np, sr = sf.read(io.BytesIO(audio_bytes))
                    else:
                        import io
                        buf = io.BytesIO(audio_bytes)
                        sr, audio_np = wavfile.read(buf)
                        audio_np = audio_np.astype(np.float32) / 32768.0
                except Exception as e:
                    continue

                if audio_np.ndim == 2:
                    audio_np = audio_np.mean(axis=1)
                audio_np = resample_audio(audio_np, sr, 16000)
                audio_np = audio_np / (np.abs(audio_np).max() + 1e-8)

                fname = f"cv21_{split_name}_{idx:06d}.wav"
                fpath = proc_dir / fname
                save_wav(str(fpath), audio_np, 16000)

                actual_dur = len(audio_np) / 16000.0

                all_records.append({
                    "audio_path": str(fpath),
                    "text": text,
                    "duration": round(actual_dur, 2),
                    "dataset": "cv21",
                    "split": split_name,
                    "speaker_id": client_id,
                })
                total_duration += actual_dur
                idx += 1

                if idx % 1000 == 0:
                    print(f"    {idx} clips processed ({total_duration/3600:.1f}h)")

    print(f"\n  CV21 total: {len(all_records)} clips, {total_duration/3600:.1f}h")
    return all_records


def process_ruls(output_dir, abbreviations):
    dataset_id = "istupakov/russian_librispeech"
    proc_dir = Path(output_dir) / "ruls"
    proc_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train": [f"data/train-{i:05d}-of-00024.parquet" for i in range(24)],
        "test": ["data/test-00000-of-00001.parquet"],
        "validation": ["data/validation-00000-of-00001.parquet"],
    }

    all_records = []
    total_duration = 0.0
    idx = 0

    for split_name, file_list in split_files.items():
        print(f"\n  Processing {split_name} split...")
        for pf in file_list:
            local = hf_hub_download(dataset_id, pf, repo_type="dataset")
            table = pq.read_table(local)
            rows = table.to_pydict()

            for i in range(table.num_rows):
                text_raw = rows["text"][i]
                if not text_raw or not text_raw.strip():
                    continue

                audio_data = rows["audio"][i]
                if audio_data is None:
                    continue

                audio_bytes = audio_data.get("bytes") or audio_data.get("path")
                if audio_bytes is None:
                    continue

                duration_s = rows["duration"][i] or 0.0

                text = normalize_text(text_raw, abbreviations)
                if not text:
                    continue

                try:
                    if HAS_SOUNDFILE:
                        import io
                        audio_np, sr = sf.read(io.BytesIO(audio_bytes))
                    else:
                        import io
                        buf = io.BytesIO(audio_bytes)
                        sr, audio_np = wavfile.read(buf)
                        audio_np = audio_np.astype(np.float32) / 32768.0
                except Exception as e:
                    continue

                if audio_np.ndim == 2:
                    audio_np = audio_np.mean(axis=1)
                audio_np = resample_audio(audio_np, sr, 16000)
                audio_np = audio_np / (np.abs(audio_np).max() + 1e-8)

                actual_dur = len(audio_np) / 16000.0
                if actual_dur < 1.0 or actual_dur > 30.0:
                    continue

                fname = f"ruls_{split_name}_{idx:06d}.wav"
                fpath = proc_dir / fname
                save_wav(str(fpath), audio_np, 16000)

                speaker = Path(rows["audio_filepath"][i]).parts[0] if rows["audio_filepath"][i] else f"spk_{idx}"

                all_records.append({
                    "audio_path": str(fpath),
                    "text": text,
                    "duration": round(actual_dur, 2),
                    "dataset": "ruls",
                    "split": split_name,
                    "speaker_id": speaker,
                })
                total_duration += actual_dur
                idx += 1

                if idx % 1000 == 0:
                    print(f"    {idx} clips processed ({total_duration/3600:.1f}h)")

    print(f"\n  RuLS total: {len(all_records)} clips, {total_duration/3600:.1f}h")
    return all_records


def split_by_speaker(records, val_ratio=0.05, test_ratio=0.0, seed=42):
    from collections import defaultdict
    import random

    random.seed(seed)
    by_speaker = defaultdict(list)
    for r in records:
        by_speaker[r["speaker_id"]].append(r)

    speakers = sorted(by_speaker.keys())
    random.shuffle(speakers)

    n_val = max(1, int(len(speakers) * val_ratio))
    n_test = max(1, int(len(speakers) * test_ratio)) if test_ratio > 0 else 0

    val_speakers = set(speakers[:n_val])
    test_speakers = set(speakers[n_val:n_val + n_test]) if n_test > 0 else set()

    train, val, test = [], [], []
    for spk in speakers:
        if spk in val_speakers:
            val.extend(by_speaker[spk])
        elif spk in test_speakers:
            test.extend(by_speaker[spk])
        else:
            train.extend(by_speaker[spk])

    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cv21", "ruls", "all"], default="all")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--manifest-dir", default="data/manifests")
    args = parser.parse_args()

    abbreviations = {}
    abbr_path = Path("data/abbreviations.json")
    if abbr_path.exists():
        with open(abbr_path) as f:
            abbreviations = json.load(f)
        print(f"Loaded {len(abbreviations)} abbreviations")

    all_records = []

    if args.dataset in ("cv21", "all"):
        print("\n=== Processing Common Voice 21 ru ===")
        records = process_cv21(args.output_dir, abbreviations)
        all_records.extend(records)

    if args.dataset in ("ruls", "all"):
        print("\n=== Processing Russian LibriSpeech ===")
        records = process_ruls(args.output_dir, abbreviations)
        all_records.extend(records)

    print(f"\n=== Splitting into train/val/test ===")
    print(f"Total records before split: {len(all_records)}")

    train, val, test = split_by_speaker(all_records, val_ratio=0.05, test_ratio=0.0)

    train_h = sum(r["duration"] for r in train) / 3600
    val_h = sum(r["duration"] for r in val) / 3600

    print(f"Train: {len(train)} clips ({train_h:.1f}h)")
    print(f"Val: {len(val)} clips ({val_h:.1f}h)")

    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("val", val)]:
        path = manifest_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                r_copy = {k: v for k, v in r.items() if k != "split"}
                f.write(json.dumps(r_copy, ensure_ascii=False) + "\n")
        print(f"Wrote {path}: {len(data)} entries")

    train_speakers = set(r["speaker_id"] for r in train)
    val_speakers = set(r["speaker_id"] for r in val)
    overlap = train_speakers & val_speakers
    print(f"Speaker overlap: {len(overlap)} speakers (should be 0)")

    versions = {
        "common_voice": {
            "version": "21.0",
            "source": "artyomboyko/common_voice_21_0_ru",
            "downloaded": datetime.now().isoformat()[:10],
            "clips": len([r for r in all_records if r["dataset"] == "cv21"]),
        },
        "russian_librispeech": {
            "source": "istupakov/russian_librispeech",
            "downloaded": datetime.now().isoformat()[:10],
            "clips": len([r for r in all_records if r["dataset"] == "ruls"]),
        },
    }
    with open("data/versions.json", "w") as f:
        json.dump(versions, f, indent=2, ensure_ascii=False)
    print(f"\nWrote data/versions.json")

    print(f"\n=== Done ===")
    print(f"Total: {len(all_records)} clips, {sum(r['duration'] for r in all_records)/3600:.1f}h")


if __name__ == "__main__":
    main()
