# SHL Hiring Assessment 2026 — Grammar Scoring Engine

> A multimodal machine learning system for automatically scoring the grammatical quality of spoken English responses.

**Final Kaggle Score: 0.33**  
**Leaderboard Rank: 15**

---

## Overview

The **SHL Hiring Assessment 2026** competition focuses on predicting the grammatical quality of spoken English responses.

Each audio recording is approximately 45–60 seconds long, and the objective is to predict a continuous **Grammar Score from 0 to 5**, corresponding to the quality of the speaker's grammatical usage.

Rather than treating this as a conventional audio regression problem, this project approaches grammar assessment as a **multimodal prediction task**.

The final system combines:

- Acoustic representations from **Whisper** and **WavLM**
- Transcript representations from **RoBERTa, DeBERTa and ELECTRA**
- Rubric-conditioned representations from **Qwen2.5-3B and Phi-3.5-mini**
- Explicit grammar measurements using **CoLA, T5-GEC, ELECTRA and GPT-2**
- Lexical and syntactic features
- Speaker-aware cross-validation
- Non-negative model stacking
- Test-time speaker-level prediction pooling

The final solution achieved a **0.33 leaderboard score and 15th place**.

---

## Key Idea

A human evaluator does not judge grammar solely from the acoustic properties of speech.

A response contains several different kinds of information:

```text
                    Spoken Response
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
          Speech       Transcript    Speaker
          Signal       / Language    Identity
             │            │
             ▼            ▼
          Whisper      Language
          WavLM       Encoders
             │            │
             └──────┬─────┘
                    │
                    ▼
             Grammar Features
                    │
          ┌─────────┼──────────┐
          ▼         ▼          ▼
        CoLA      T5-GEC     GPT-2
        Syntax    ELECTRA    Surprisal
                    │
                    ▼
              LLM Examiner
             Qwen / Phi
                    │
                    ▼
            Multimodal Models
                    │
                    ▼
              Model Stacking
                    │
                    ▼
             Final Prediction
