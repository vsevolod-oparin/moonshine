# Russian Speech Datasets — Availability Tracker

Living document. Updated as datasets are discovered, verified, or downloaded.

## Datasets Currently in Pipeline

| Dataset | Hours | Status | Location |
|---------|-------|--------|----------|
| Common Voice 21 ru (`artyomboyko/common_voice_21_0_ru`) | ~170h | Downloaded, processed | `data/processed/cv21/` |
| Russian LibriSpeech (`istupakov/russian_librispeech`) | ~98h | Downloaded, processed | `data/processed/ruls/` |
| **Total in pipeline** | **~248h** | | |

---

## Available on HuggingFace (verified, with data files)

| # | Dataset | HF ID | Hours | Rows | License | Audio | Quality | Notes |
|---|---------|-------|-------|------|---------|-------|---------|-------|
| 1 | **SOVA RuDevices** | `bond005/sova_rudevices` | ~97h | ~82K | CC-BY-4.0 | WAV 16kHz | Spontaneous speech, manual annotation | Conversational/spontaneous — high value for real-world ASR |
| 2 | **SOVA Audiobooks** | `dangrebenkin/sova_rudevices_audiobooks` | ~338h | ~183K | CC-BY-4.0 | WAV | Audiobook read speech | Large, clean read speech |
| 3 | **Taiga Speech** | `bond005/taiga_speech` | ~857h | ~174K | Unknown | WAV | Read audiobook, 5 speakers | Large but few speakers. Some clips >30s need splitting |
| 4 | **Taiga Speech v2** | `bond005/taiga_speech_v2` | ~34h | ~12K | Unknown | WAV | 2 speakers | Small, clean |
| 5 | **ToneWebinars** | `Vikhrmodels/ToneWebinars` | ~2,020h | ~287K | Apache-2.0 | MP3 48kHz | Educational webinars, auto-transcribed | Massive. Avg clip 25s. Quality depends on auto-transcription |
| 6 | **ESpeech-Webinars2** | `ESpeech/ESpeech-webinars2` | ~850h | N/A | Apache-2.0 | MP3 44.1kHz | Webinar speech, auto-transcribed | Has word-level timestamps, speaker IDs, quality scores. 74GB tar split |
| 7 | **Golos crowd 10h** | `bond005/sberdevices_golos_10h_crowd` | ~21h | ~19K | SberDevices | WAV 16kHz | Crowd-sourced, close-talk | Subset of full Golos |
| 8 | **Golos farfield 100h** | `bond005/sberdevices_golos_100h_farfield` | ~13h | ~12K | SberDevices | WAV 16kHz | Far-field recordings | Subset of full Golos |
| 9 | **Golos crowd noised** | `bond005/sberdevices_golos_10h_crowd_noised_2db` | ~21h | ~18K | SberDevices | WAV 16kHz | Noise-augmented crowd | Same as crowd 10h + noise |
| 10 | **Golos (community mirror)** | `gggggggg123/sber-golos` | unknown | ~19K | Unknown | Unknown | Whisper-format features | Pre-extracted features, not raw audio |
| 11 | **Golos Balalaika** | `lab260/golos_balalaika` | unknown | unknown | Unknown | Unknown | Subset of Golos | Structure unclear |
| 12 | **CV+SOVA+Golos+Fleurs merged** | `Arciomwww/ru_common_voice_sova_rudevices_golos_fleurs` | ~83h | ~142K | Mixed | Mixed | Merged from multiple sources | Pre-merged, may have overlap with individual datasets |
| 13 | **Common Voice 22** | `mort666/cv_corpus_v22` (subset `ru`) | ~100-120h | ~75K | CC0 | MP3 48kHz | Crowd-validated | Community mirror. Newer than CV21 |
| 14 | **Common Voice 21** | `artyomboyko/common_voice_21_0_ru` | ~242h | ~170K | CC0 | Embedded | Crowd-validated | **Already in pipeline** |
| 15 | **Russian single speaker** | `niobures/russian-single-speaker-speech-dataset` | small | ~19K | Unknown | WAV | Single speaker | 19K files but single speaker, limited value |
| 16 | **Podlodka Speech** | `bond005/podlodka_speech` | ~1.5h | 107 | Unknown | WAV | Podcast clips | Too small |
| 17 | **Noise-augmented RuLS** | `FacelessLake/noise-augmented-russian-librispeech` | ~98h | unknown | Unknown | WAV | RuLS + noise | Augmented version of RuLS, may be useful for robustness |

## Available but Not on HuggingFace

| # | Dataset | Source | Hours | License | Notes |
|---|---------|--------|-------|---------|-------|
| 1 | **Golos full corpus** | OpenSLR (SLR60) | ~1,200h | CC-BY-4.0 | Main Golos corpus. HF mirrors only have small subsets. Full download from OpenSLR |
| 2 | **Common Voice latest** | Mozilla Data Collective | ~3,000h+ (all versions) | CC0 | Moved off HF Oct 2025. Requires license acceptance at datacollective.mozillafoundation.org |
| 3 | **OpenSTT / Balalaika** | `lab260/openstt_balalaika` on HF (metadata only) | ~14,000h | **CC-BY-NC-4.0** | Phone calls. Metadata on HF, audio from OpenSTT archive. **Non-commercial only** |
| 4 | **MUSAN** | OpenSLR (SLR17) | ~60h noise/music/babble | Free for research | Noise augmentation corpus. Need to request access or find mirror |

## Not Available / Empty / Broken

| Dataset | Status |
|---------|--------|
| `SberDevices/Golos` | Empty — only README + .gitattributes |
| `mozilla-foundation/common_voice_17_0` | Empty — data removed Oct 2025 |
| `jims57/cv-corpus-20` | Empty |
| `i-koskin/Golos` | Empty |
| `facebook/multilingual_librispeech` (Russian) | No Russian config exists |
| `salute-developers/golos` | 404/gated |

---

## Recommended Priority for Pipeline Expansion

| Priority | Dataset | Add Hours | Effort | Why |
|----------|---------|-----------|--------|-----|
| **1** | SOVA RuDevices (`bond005/sova_rudevices`) | +97h | Easy — HF, same format | Spontaneous speech adds diversity, CC-BY-4.0 |
| **2** | SOVA Audiobooks (`dangrebenkin/sova_rudevices_audiobooks`) | +338h | Easy — HF | Large clean read speech, CC-BY-4.0 |
| **3** | Common Voice 22 (`mort666/cv_corpus_v22`) | +100h | Medium — MP3, 48kHz | More diverse speakers than CV21 |
| **4** | Golos full (OpenSLR SLR60) | +1,200h | Hard — manual download | Single biggest gain. Far-field data |
| **5** | ToneWebinars (`Vikhrmodels/ToneWebinars`) | +2,020h | Medium — 352 shards, MP3 | Massive, Apache-2.0, but auto-transcribed (quality?) |
| **6** | ESpeech-Webinars2 (`ESpeech/ESpeech-webinars2`) | +850h | Hard — 74GB tar split, MP3 | Apache-2.0, has quality scores for filtering |
| **7** | Taiga Speech (`bond005/taiga_speech`) | +857h | Medium — 249 shards | Large but only 5 speakers, may overfit |
| **8** | OpenSTT Balalaika | +14,000h | Very hard — NC license | Non-commercial only. For research Phase 3 |

**Potential total (commercial-safe, items 1-7): ~4,800h+**
**Potential total (research, including OpenSTT): ~18,800h+**

---

## Version History

- **2026-04-26**: Initial creation. CV21 (170h) + RuLS (98h) = 248h in pipeline. Researched 110+ HF datasets, verified 17 with data.
