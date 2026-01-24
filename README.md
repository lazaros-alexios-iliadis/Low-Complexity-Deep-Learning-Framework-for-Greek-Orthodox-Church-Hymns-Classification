This repository contains the code accompanying the paper  
**“Low Complexity Deep Learning Framework for Greek Orthodox Church Hymns Classification”**  
published in *Applied Sciences*.  
DOI: https://doi.org/10.3390/app13158638

The work investigates **deep learning–based audio classification** with a strong emphasis on **low computational complexity**, making the proposed methods suitable for resource-constrained or embedded systems.

---

## Problem Description

Automatic classification of Greek Orthodox Church hymns presents challenges related to:
- Subtle spectral differences between classes
- Limited availability of labeled audio data
- The need for computationally efficient models

This project addresses these challenges by combining **signal processing techniques** with **lightweight deep learning architectures**, focusing on performance–complexity trade-offs.

---

## Methodology

The implemented pipeline consists of the following steps:

### 1. Audio Preprocessing
- Conversion of audio signals into **Mel-spectrogram representations**
- Time–frequency analysis to capture relevant harmonic and temporal features

### 2. Deep Learning Models
- **Three custom-designed CNN architectures**, optimized for low complexity
- **Five pre-trained CNN models** used for performance benchmarking
- All models trained and evaluated on a private dataset of Byzantine music recordings

### 3. Evaluation
- Comparative analysis based on:
  - Classification accuracy
  - Model complexity
  - Computational cost

This allows a systematic assessment of the trade-off between model performance and resource requirements.

---

## Implementation Details

- **Language:** Python  
- **Framework:** PyTorch / TensorFlow (depending on implementation)  
- **Input Representation:** Mel-spectrograms  
- **Data:** Private audio dataset (not included)

Due to copyright and licensing restrictions, the dataset cannot be publicly distributed. The provided code documents the full methodology and can be adapted to similar audio classification tasks.

---

## Research Context

This research was conducted as part of the project:

**“Recognition and direct characterization of cultural items for the education and promotion of Byzantine Music using artificial intelligence”**  
(Project code: **KMP6-0078938**)

Funded under the Action *“Investment Plans of Innovation”* of the Operational Program *Central Macedonia 2014–2020*, co-funded by the **European Regional Development Fund (ERDF)** and **Greece**.

---

## Notes

- The focus of this repository is on **efficient model design**, not solely on maximizing accuracy.
- The proposed framework highlights the importance of **signal-aware preprocessing** and **model complexity control**.
- The results are directly applicable to embedded or real-time audio classification scenarios.

---
