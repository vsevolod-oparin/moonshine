import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sentencepiece as spm


def pretokenize(manifest_path: str, tokenizer_path: str, output_path: str):
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    count = 0
    with open(manifest_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            token_ids = sp.encode(record["text"], out_type=int)
            token_ids = [t for t in token_ids if t >= 6]
            record["token_ids"] = token_ids
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"  {count:,} records...", flush=True)

    print(f"Done: {count:,} records → {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize manifest files")
    parser.add_argument("--manifest", required=True, help="Input manifest path")
    parser.add_argument("--tokenizer", default="data/tokenizer_256.model")
    parser.add_argument("--output", help="Output path (default: overwrite input)")
    args = parser.parse_args()

    output = args.output or args.manifest
    pretokenize(args.manifest, args.tokenizer, output)


if __name__ == "__main__":
    main()
