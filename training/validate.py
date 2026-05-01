import torch
from typing import Optional

import jiwer


_SPECIAL_IDS = {0, 3, 4, 5}
_EOS_ID = 2


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


def aed_greedy_decode(
    model,
    enc_output: torch.Tensor,
    enc_lengths: torch.Tensor | None = None,
    max_len: int | torch.Tensor | None = 448,
) -> list[list[int]]:
    device = enc_output.device
    batch_size = enc_output.size(0)

    tokens = torch.full((batch_size, 1), model.config.sos_eos_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    results: list[list[int]] = [[] for _ in range(batch_size)]

    if isinstance(max_len, torch.Tensor):
        max_per_sample = max_len.int().tolist()
    elif max_len is None:
        max_per_sample = [448] * batch_size
    else:
        max_per_sample = [max_len] * batch_size
    global_max = max(max_per_sample)

    for step_i in range(global_max):
        logits = model.decode(tokens, enc_output, enc_lengths)
        next_token = logits[:, -1, :].argmax(dim=-1)

        newly_finished = (next_token == _EOS_ID) | (next_token == model.config.pad_token_id)
        for i in range(batch_size):
            if not finished[i]:
                if step_i >= max_per_sample[i]:
                    finished[i] = True
                else:
                    t = next_token[i].item()
                    if newly_finished[i]:
                        finished[i] = True
                    elif t not in _SPECIAL_IDS:
                        results[i].append(t)

        if finished.all():
            break

        tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=-1)

    return results


@torch.inference_mode()
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
    stats_ctc = ErrorRateStats()
    stats_aed = ErrorRateStats()
    total_loss = 0.0
    total_batches = 0
    all_refs_ctc = []
    all_hyps_ctc = []
    all_refs_aed = []
    all_hyps_aed = []

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

        # CTC greedy decode
        ctc_ids = ctc_greedy_decode(ctc_logits, enc_lengths, model.config.blank_token_id)
        hyps_ctc, refs_ctc = _decode_texts(ctc_ids, tokens, token_lengths, tokenizer)
        stats_ctc.update(refs_ctc, hyps_ctc)
        all_refs_ctc.extend(refs_ctc)
        all_hyps_ctc.extend(hyps_ctc)

        aed_ids = aed_greedy_decode(model, enc_output, enc_lengths, max_len=448)
        hyps_aed, refs_aed = _decode_texts(aed_ids, tokens, token_lengths, tokenizer)
        stats_aed.update(refs_aed, hyps_aed)
        all_refs_aed.extend(refs_aed)
        all_hyps_aed.extend(hyps_aed)

    model.train()

    result = stats_ctc.summary()
    result["val_loss"] = total_loss / max(total_batches, 1)
    result["cer"] = jiwer.cer(all_refs_ctc, all_hyps_ctc) * 100 if all_refs_ctc else 0.0
    result["wer_aed"] = stats_aed.wer
    result["ser_aed"] = stats_aed.ser
    result["cer_aed"] = jiwer.cer(all_refs_aed, all_hyps_aed) * 100 if all_refs_aed else 0.0
    return result


def _decode_texts(
    token_ids_list: list[list[int]],
    tokens: torch.Tensor,
    token_lengths: torch.Tensor,
    tokenizer,
) -> tuple[list[str], list[str]]:
    hyps: list[str] = []
    refs: list[str] = []
    for i, ids in enumerate(token_ids_list):
        hyp_ids = [t for t in ids if t >= 6]
        hyp_text = tokenizer.decode(hyp_ids) if hyp_ids else ""
        ref_len = token_lengths[i].item()
        ref_ids = tokens[i, :ref_len].tolist()
        ref_ids = [t for t in ref_ids if t >= 6]
        ref_text = tokenizer.decode(ref_ids) if ref_ids else ""
        hyps.append(hyp_text)
        refs.append(ref_text)
    return hyps, refs
