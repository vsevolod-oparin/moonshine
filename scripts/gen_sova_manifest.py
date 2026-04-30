import io
import json
import re
import hashlib
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq
import soundfile as sf


def normalize_text(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[«»\"\u201e\u201c\u201d␤]', "", text)
    text = re.sub(r"[\(\)\[\]{}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\sа-яёА-ЯЁ\-.,;:!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_sova(dataset_id, dataset_name):
    proc_dir = Path(f"data/processed/{dataset_name}")
    files = list_repo_files(dataset_id, repo_type="dataset")
    parquets = sorted([f for f in files if f.startswith("data/") and f.endswith(".parquet")])

    records = []
    idx = 0
    for pf in parquets:
        split_name = (
            "train" if "train" in pf else ("test" if "test" in pf else "validation")
        )
        print(f"  {pf} ({split_name})...", flush=True)
        local = hf_hub_download(dataset_id, pf, repo_type="dataset")
        table = pq.read_table(local)
        rows = table.to_pydict()

        for i in range(table.num_rows):
            text_raw = rows.get("transcription", rows.get("text", [""]))[i]
            if not text_raw or not text_raw.strip():
                continue

            audio_data = rows["audio"][i]
            if audio_data is None:
                continue
            audio_bytes = audio_data.get("bytes")
            if audio_bytes is None:
                continue

            text = normalize_text(text_raw)
            if not text:
                continue

            try:
                audio_np, sr = sf.read(io.BytesIO(audio_bytes))
            except Exception:
                continue

            if audio_np.ndim == 2:
                audio_np = audio_np.mean(axis=1)
            dur = len(audio_np) / sr

            if dur < 1.0 or dur > 30.0:
                continue

            fname = f"{dataset_name}_{split_name}_{idx:06d}.wav"
            fpath = proc_dir / fname

            content_hash = hashlib.md5(audio_bytes).hexdigest()[:8]

            records.append(
                {
                    "audio_path": str(fpath),
                    "text": text,
                    "duration": round(dur, 2),
                    "dataset": dataset_name,
                    "speaker_id": f"{dataset_name}_{content_hash}",
                }
            )
            idx += 1

            if idx % 10000 == 0:
                print(f"    {idx} records...", flush=True)

    return records


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "sova_rudevices"

    if dataset == "sova_rudevices":
        print("=== SOVA RuDevices ===")
        records = process_sova("bond005/sova_rudevices", "sova_rudevices")
    elif dataset == "sova_audiobooks":
        print("=== SOVA Audiobooks ===")
        records = process_sova(
            "dangrebenkin/sova_rudevices_audiobooks", "sova_audiobooks"
        )
    else:
        print(f"Unknown dataset: {dataset}")
        sys.exit(1)

    total_h = sum(r["duration"] for r in records) / 3600
    print(f"Total: {len(records)} records ({total_h:.1f}h)")

    out_path = f"data/manifests/{dataset}_raw.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")
