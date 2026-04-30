import torch
from typing import Optional

import jiwer


class ErrorRateStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_insertions = 0
        self.total_deletions = 0
        self.total_substitutions = 0
        self.total_ref_words = 0
        self.num_utterances = 0
        self.num_errors = 0

    def update(self, references: list[str], hypotheses: list[str]):
        for ref, hyp in zip(references, hypotheses):
            ref_words = ref.split()

            out = jiwer.process_words(ref, hyp)
            self.total_insertions += out.insertions
            self.total_deletions += out.deletions
            self.total_substitutions += out.substitutions
            self.total_ref_words += len(ref_words)

            if ref != hyp:
                self.num_errors += 1
            self.num_utterances += 1

    @property
    def wer(self) -> float:
        if self.total_ref_words == 0:
            return 0.0
        ops = self.total_insertions + self.total_deletions + self.total_substitutions
        return 100.0 * ops / self.total_ref_words

    @property
    def ser(self) -> float:
        if self.num_utterances == 0:
            return 0.0
        return 100.0 * self.num_errors / self.num_utterances

    def summary(self) -> dict[str, float]:
        return {
            "wer": self.wer,
            "ser": self.ser,
            "total_ref_words": self.total_ref_words,
            "total_utterances": self.num_utterances,
        }


def ctc_greedy_decode(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_id: int,
) -> list[list[int]]:
    argmax = logits.argmax(dim=-1)

    results = []
    for i in range(argmax.size(0)):
        length = input_lengths[i].item()
        tokens = argmax[i, :length].tolist()

        collapsed = []
        prev = blank_id
        for t in tokens:
            if t != blank_id and t != prev:
                collapsed.append(t)
            prev = t
        results.append(collapsed)
    return results


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader,
    tokenizer,
    device: torch.device,
    max_batches: Optional[int] = None,
    precision: str = "fp32",
) -> dict:
    model.eval()
    use_autocast = precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    stats = ErrorRateStats()
    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        audio, audio_lengths, tokens, token_lengths = batch
        audio = audio.to(device)
        audio_lengths = audio_lengths.to(device)
        tokens = tokens.to(device)
        token_lengths = token_lengths.to(device)

        with torch.amp.autocast("cuda", enabled=use_autocast, dtype=amp_dtype):
            enc_output, enc_lengths = model.encode(audio, audio_lengths)
            ctc_logits = model.ctc_head(enc_output)

            ctc_log_probs = torch.nn.functional.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
        if enc_lengths is None:
            enc_lengths = torch.full(
                (enc_output.size(0),),
                enc_output.size(1),
                dtype=torch.long,
                device=device,
            )
        loss_ctc = torch.nn.functional.ctc_loss(
            ctc_log_probs, tokens, enc_lengths, token_lengths,
            blank=model.config.blank_token_id,
            reduction="mean", zero_infinity=True,
        )
        total_loss += loss_ctc.item()
        total_batches += 1

        decoded_token_ids = ctc_greedy_decode(
            ctc_logits, enc_lengths, model.config.blank_token_id
        )

        hypotheses = []
        references = []
        for i, ids in enumerate(decoded_token_ids):
            hyp_ids = [t for t in ids if t >= 6]
            hyp_text = tokenizer.decode(hyp_ids) if hyp_ids else ""
            ref_len = token_lengths[i].item()
            ref_ids = tokens[i, :ref_len].tolist()
            ref_ids = [t for t in ref_ids if t >= 6]
            ref_text = tokenizer.decode(ref_ids) if ref_ids else ""
            hypotheses.append(hyp_text)
            references.append(ref_text)

        stats.update(references, hypotheses)

    model.train()

    result = stats.summary()
    result["val_loss"] = total_loss / max(total_batches, 1)
    return result
