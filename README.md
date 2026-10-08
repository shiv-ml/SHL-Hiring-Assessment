# SHL Hiring Assessment 2026 — Grammar Scoring Engine

A multimodal machine learning system for scoring the grammatical quality of spoken English from 45–60 second audio recordings.

**Kaggle Result: Rank 15 | Score: 0.33**

## Overview

The task was to predict a continuous grammar score from **0 to 5** for spoken-audio samples.

The final solution treated grammar scoring as a **multimodal regression problem**, combining:

- Speech representations from **Whisper / Distil-Whisper** and **WavLM**
- Transcript representations from **RoBERTa, DeBERTa and ELECTRA**
- Rubric-conditioned features from **Qwen2.5-3B** and **Phi-3.5-mini**
- Explicit grammar features using **CoLA, T5-GEC and GPT-2 surprisal**
- Syntactic, lexical and fluency features
- Speaker embeddings for leakage-aware validation
- Ridge, RBF-SVR and LightGBM regression
- Non-negative stacking
- Test-time speaker pooling

## Architecture

```text
Audio
  │
  ├── Whisper / Distil-Whisper ──→ Transcript
  │                                  │
  └── WavLM ──→ Acoustic Features   │
                                     │
             ┌───────────────────────┼──────────────────────┐
             │                       │                      │
             ▼                       ▼                      ▼
        RoBERTa /              Grammar Features       Qwen / Phi
        DeBERTa /              CoLA / T5-GEC /       LLM Examiner
        ELECTRA                GPT-2 / Syntax
             │                       │                      │
             └───────────────────────┼──────────────────────┘
                                     │
                                     ▼
                              Model Ensemble
                                     │
                         ┌───────────┼───────────┐
                         ▼           ▼           ▼
                       Ridge      RBF-SVR    LightGBM
                         └───────────┼───────────┘
                                     ▼
                            Non-negative Stacking
                                     │
                                     ▼
                              Final Score (0–5)
```

## Key Methodology

### Multimodal representations

Different pretrained models capture different aspects of spoken language. Audio models provide speech-level information, while text models capture linguistic and grammatical patterns from the transcript.

### Explicit grammar measurements

The system also extracted interpretable signals related to grammatical acceptability, correction requirements, surprisal, syntax, lexical richness and fluency.

### Speaker-aware validation

Speaker embeddings were used to identify repeated speakers. This was important because random cross-validation can place recordings from the same speaker in both training and validation sets, producing overly optimistic estimates.

Speaker-aware grouping was therefore used during model evaluation.

### Ensemble learning

Multiple regression models were trained on complementary feature views and combined using non-negative stacking. This allowed the final predictor to benefit from different model strengths rather than relying on a single representation.

## Results

| Metric | Result |
|---|---:|
| Kaggle Score | **0.33** |
| Kaggle Rank | **15** |
| Training Samples | 769 |


## Key Takeaways

- Grammar scoring benefits from combining **audio and text representations**.
- Explicit linguistic features can complement large pretrained models.
- LLMs can be useful as **feature extractors and rubric-based examiners**, not only as final predictors.
- Speaker-aware validation is essential when recordings from the same speaker occur multiple times.
- On a relatively small dataset, combining strong pretrained representations with lightweight regression models can be more practical than end-to-end fine-tuning.

| Test Samples | 216 |
| Target | Grammar Score (0–5) |
