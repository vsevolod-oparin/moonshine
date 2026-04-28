"""PyTorch Dataset for Russian ASR training.

Reads manifests, loads audio, applies configurable augmentation.
"""

import json
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def load_manifest(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class AudioProcessor:
    MEL_PARAMS = {
        "sample_rate": 16000,
        "n_fft": 512,
        "win_length": 400,
        "hop_length": 160,
        "n_mels": 80,
        "f_min": 0,
        "f_max": 8000,
    }

    def __init__(self, mel_params: Optional[dict] = None):
        params = {**self.MEL_PARAMS, **(mel_params or {})}
        self.sample_rate = params["sample_rate"]
        self.n_fft = params["n_fft"]
        self.win_length = params["win_length"]
        self.hop_length = params["hop_length"]
        self.n_mels = params["n_mels"]
        self._f_min = params["f_min"]
        self._f_max = params["f_max"]

        try:
            import torchaudio
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=params["f_min"],
                f_max=params["f_max"],
            )
            self._amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
            self._has_ta = True
        except Exception:
            self._has_ta = False

    def load_audio(self, path: str) -> np.ndarray:
        try:
            import soundfile as sf
            audio, sr = sf.read(path)
        except ImportError:
            import scipy.io.wavfile as wavfile
            sr, audio = wavfile.read(path)
            audio = audio.astype(np.float32) / 32768.0

        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(self.sample_rate, sr)
            audio = resample_poly(audio, self.sample_rate // g, sr // g).astype(np.float32)

        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio.astype(np.float32)

    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        if self._has_ta:
            mel = self._mel_transform(audio)
            mel = self._amp_to_db(mel)
            mel = (mel + 80) / 40.0
            return mel.squeeze(0)
        else:
            window = torch.hann_window(self.win_length)
            stft = torch.stft(
                audio, self.n_fft, self.hop_length, self.win_length, window,
                return_complex=True,
            )
            mag = stft.abs() ** 2
            n_freq_bins = mag.shape[-2]
            f_min_hz = getattr(self, '_f_min', 0) or 0
            f_max_hz = getattr(self, '_f_max', 8000) or 8000

            def hz_to_mel(hz):
                return 2595.0 * torch.log10(torch.tensor(1.0) + hz / 700.0)

            def mel_to_hz(mel):
                return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

            mel_min = hz_to_mel(f_min_hz)
            mel_max = hz_to_mel(f_max_hz)
            mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
            hz_points = mel_to_hz(mel_points)

            bin_points = ((f_max_hz - f_min_hz) / n_freq_bins) * torch.arange(n_freq_bins).float() + f_min_hz

            fbank = torch.zeros(self.n_mels, n_freq_bins)
            for m in range(self.n_mels):
                f_left = hz_points[m]
                f_center = hz_points[m + 1]
                f_right = hz_points[m + 2]
                for k in range(n_freq_bins):
                    f = bin_points[k]
                    if f_left <= f <= f_center and f_center > f_left:
                        fbank[m, k] = (f - f_left) / (f_center - f_left)
                    elif f_center < f <= f_right and f_right > f_center:
                        fbank[m, k] = (f_right - f) / (f_right - f_center)

            mel = torch.matmul(fbank, mag.squeeze(0))
            mel = 10.0 * torch.log10(mel + 1e-10)
            mel = (mel + 80) / 40.0
            return mel


class SpecAugment:
    def __init__(self, freq_mask=15, time_mask=50, n_freq_masks=2, n_time_masks=2):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        _, n_mels, n_frames = mel.shape if mel.dim() == 3 else (1,) + mel.shape
        mel = mel.unsqueeze(0) if mel.dim() == 2 else mel

        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask)
            f0 = random.randint(0, max(0, n_mels - f))
            mel[:, f0:f0 + f, :] = 0

        for _ in range(self.n_time_masks):
            t = random.randint(0, self.time_mask)
            t0 = random.randint(0, max(0, n_frames - t))
            mel[:, :, t0:t0 + t] = 0

        return mel.squeeze(0)


class SpeedPerturbation:
    SPEEDS = [0.9, 1.0, 1.1]

    def __init__(self, speeds: list[float] = None):
        self.speeds = speeds or self.SPEEDS

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        speed = random.choice(self.speeds)
        if speed == 1.0:
            return audio
        n_samples = int(len(audio) / speed)
        indices = np.linspace(0, len(audio) - 1, n_samples).astype(int)
        return audio[indices]


class ASRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        audio_processor: Optional[AudioProcessor] = None,
        tokenizer_model: Optional[str] = None,
        max_duration: float = 30.0,
        min_duration: float = 1.0,
        spec_augment: bool = False,
        speed_perturbation: bool = False,
        raw_audio: bool = False,
    ):
        self.records = load_manifest(manifest_path)
        self.processor = audio_processor or AudioProcessor()
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.raw_audio = raw_audio

        self.records = [
            r for r in self.records
            if self.min_duration <= r["duration"] <= self.max_duration
        ]

        self.spec_augment = SpecAugment() if spec_augment else None
        self.speed_perturb = SpeedPerturbation() if speed_perturbation else None

        self._tokenizer_path = tokenizer_model
        self.tokenizer = None

    def __len__(self):
        return len(self.records)

    def _ensure_tokenizer(self):
        if self.tokenizer is None and self._tokenizer_path:
            import sentencepiece as spm

            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.Load(self._tokenizer_path)

    def __getitem__(self, idx):
        self._ensure_tokenizer()
        record = self.records[idx]

        audio = self.processor.load_audio(record["audio_path"])

        if self.speed_perturb:
            audio = self.speed_perturb(audio)

        if "token_ids" in record:
            token_ids = [t for t in record["token_ids"] if t >= 6]
            tokens = torch.tensor(token_ids, dtype=torch.long)
        elif self.tokenizer:
            token_ids = self.tokenizer.encode(record["text"], out_type=int)
            token_ids = [t for t in token_ids if t >= 6]
            tokens = torch.tensor(token_ids, dtype=torch.long)
        else:
            tokens = record["text"]

        if self.raw_audio:
            return torch.from_numpy(audio).float(), tokens

        mel = self.processor.audio_to_mel(torch.from_numpy(audio))

        if self.spec_augment:
            mel = self.spec_augment(mel)

        return mel, tokens


def collate_fn(batch):
    inputs, texts = zip(*batch)

    is_raw_audio = inputs[0].dim() == 1

    if is_raw_audio:
        audio_lengths = [a.shape[0] for a in inputs]
        max_audio_len = max(audio_lengths)
        audio_padded = []
        for a in inputs:
            pad_size = max_audio_len - a.shape[0]
            audio_padded.append(F.pad(a, (0, pad_size)))
        audio_batch = torch.stack(audio_padded)
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)
    else:
        mel_lengths = [m.shape[-1] for m in inputs]
        max_mel_len = max(mel_lengths)
        mels_padded = []
        for mel in inputs:
            if mel.dim() == 2:
                pad_size = max_mel_len - mel.shape[-1]
                mels_padded.append(F.pad(mel, (0, pad_size)))
            else:
                mels_padded.append(mel)
        audio_batch = torch.stack(mels_padded)
        audio_lengths = torch.tensor(mel_lengths, dtype=torch.long)

    if isinstance(texts[0], torch.Tensor):
        max_text_len = max(t.shape[0] for t in texts)
        texts_padded = []
        text_lengths = []
        for t in texts:
            pad_size = max_text_len - t.shape[0]
            texts_padded.append(F.pad(t, (0, pad_size), value=-100))
            text_lengths.append(t.shape[0])
        texts_batch = torch.stack(texts_padded)
        text_lengths = torch.tensor(text_lengths, dtype=torch.long)
        return audio_batch, audio_lengths, texts_batch, text_lengths

    return audio_batch, audio_lengths, texts


def create_dataloader(
    manifest_path: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    tokenizer_model: Optional[str] = None,
    spec_augment: bool = False,
    speed_perturbation: bool = False,
    raw_audio: bool = False,
):
    dataset = ASRDataset(
        manifest_path=manifest_path,
        tokenizer_model=tokenizer_model,
        spec_augment=spec_augment,
        speed_perturbation=speed_perturbation,
        raw_audio=raw_audio,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    return loader
