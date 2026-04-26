"""Train SentencePiece BPE tokenizers for Russian (vocab 256, 512, 1024)."""

import sentencepiece as spm
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
CORPUS = str(DATA_DIR / "tokenizer_corpus.txt")

DIGITS = [str(i) for i in range(10)]
USER_SYMBOLS = ["<blank>", "<sos/eos>", "<pad>", "<COMMA>", "<PERIOD>", "<QUESTION>"] + DIGITS

for vocab_size in [256, 512, 1024]:
    prefix = str(DATA_DIR / f"tokenizer_{vocab_size}")
    print(f"Training vocab_size={vocab_size}...")
    spm.SentencePieceTrainer.train(
        input=CORPUS,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        split_digits=True,
        split_by_unicode_script=False,
        max_sentencepiece_length=24,
        user_defined_symbols=USER_SYMBOLS,
    )
    print(f"  -> {prefix}.model, {prefix}.vocab")

print("Done.")
