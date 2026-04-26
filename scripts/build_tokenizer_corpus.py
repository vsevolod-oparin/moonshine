"""Download transcript text from Russian speech datasets for tokenizer training.

Downloads only text columns (no audio) from parquet files on HuggingFace.
"""

import os
import sys
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
CORPUS_PATH = DATA_DIR / "tokenizer_corpus.txt"


def download_parquet_texts(dataset_id: str, parquet_files: list[str], text_column: str) -> list[str]:
    texts = []
    for pf in parquet_files:
        local = hf_hub_download(dataset_id, pf, repo_type="dataset")
        table = pq.read_table(local, columns=[text_column])
        col = table.column(text_column)
        for val in col:
            s = val.as_py()
            if s and s.strip():
                texts.append(s.strip())
        print(f"  {pf}: {len(col)} rows")
    return texts


def normalize_for_tokenizer(texts: list[str]) -> list[str]:
    import re
    normalized = []
    for t in texts:
        t = t.lower()
        t = re.sub(r'[^\w\sа-яёА-ЯЁ\-]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        if t:
            normalized.append(t)
    return normalized


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_texts = []

    # --- Common Voice 21 ru (validated split = 122K sentences) ---
    print("Downloading Common Voice 21 ru transcripts...")
    cv_files = [f"data/validated-{i:05d}-of-00010.parquet" for i in range(10)]
    cv_texts = download_parquet_texts("artyomboyko/common_voice_21_0_ru", cv_files, "sentence")
    all_texts.extend(cv_texts)
    print(f"  CV21 total: {len(cv_texts)} sentences")

    # --- Russian LibriSpeech (~98 hours) ---
    print("Downloading Russian LibriSpeech transcripts...")
    ruls_files = [f"data/train-{i:05d}-of-00024.parquet" for i in range(24)]
    ruls_files.append("data/test-00000-of-00001.parquet")
    ruls_files.append("data/validation-00000-of-00001.parquet")
    ruls_texts = download_parquet_texts("istupakov/russian_librispeech", ruls_files, "text")
    all_texts.extend(ruls_texts)
    print(f"  RuLS total: {len(ruls_texts)} sentences")

    # --- Deduplicate ---
    before = len(all_texts)
    all_texts = list(dict.fromkeys(all_texts))
    print(f"Deduplicated: {before} -> {len(all_texts)}")

    # --- Normalize ---
    all_texts = normalize_for_tokenizer(all_texts)
    print(f"After normalization: {len(all_texts)} sentences")

    # --- Write corpus ---
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for t in all_texts:
            f.write(t + "\n")
    print(f"Written to {CORPUS_PATH}: {len(all_texts)} lines")


if __name__ == "__main__":
    main()
