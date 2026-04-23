# GLM Comments on ru-Moonshine Plan

## Tokenizer: Reuse vs Train Custom

### Existing Tokenizers Available

| Option | Vocab | Embedding params (dim=512) | Pros | Cons |
|--------|-------|---------------------------|------|------|
| **Whisper multilingual BPE** | 51,865 | 26.5M | Battle-tested, handles Russian, off-the-shelf | 98% of tokens are for other languages (waste). Large decoder. |
| **Train custom Russian BPE** | 256-1,024 | 0.1-0.5M | Small, fast decoding, optimized for Russian | Need to train it (trivial — minutes on text data) |
| **T-one character alphabet** | ~35 | 18K | Simplest, zero OOV | 5× more decoder steps per word |

### Recommendation: Train Custom

Whisper's tokenizer is the obvious reuse candidate — it already handles Russian and is available via `whisper.tokenizer.get_tokenizer(multilingual=True)`. But 51,865 tokens means the decoder embedding matrix alone is 26.5M params (21% of Small's total 123M budget) — mostly wasted on Chinese characters and Korean syllables that will never be used.

Training your own is ~10 lines of code and 2 minutes of compute:

```python
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='combined_russian_transcripts.txt',
    model_prefix='ru_bpe',
    vocab_size=1024,
    model_type='bpe',
    character_coverage=1.0,
)
```

The transcripts from training data (Common Voice + Golos + MLS) serve as tokenizer training data — already downloaded, no extra work.

**One case to reuse Whisper's tokenizer**: if you want to fine-tune Whisper or use Whisper as a teacher for pseudo-labeling and need token-level compatibility. Since we're training from scratch with Moonshine architecture, there's no benefit.
