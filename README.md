# HMM Speech Emotion Recognition

Reproduction of an HMM-based speech emotion recognition system with integrated MFCC, pitch, and energy features. Built for CS5100 FAI Capstone, Spring 2026.

- Report: report.pdf (included in this repository)

---

## Overview

This project implements a Hidden Markov Model (HMM) classifier for four emotion classes: happy, sad, angry, neutral. Each class is modeled by a 5-state left-to-right HMM with full-covariance Gaussian emissions, trained using the Baum-Welch (EM) algorithm.

Features are extracted per audio frame as a 41-dimensional vector:
- 13 MFCC coefficients
- 13 delta (first-order) coefficients
- 13 delta-delta (second-order) coefficients
- Normalized pitch (scaled by 0.3)
- Normalized log energy (scaled by 0.3)

Classification selects the class with the highest log-likelihood:

c_hat = argmax_c log P(O | lambda_c)

A detailed analysis of model behavior and evaluation of HMM assumptions is provided in report.pdf.

---

## Repository Structure

hmm-speech-emotion/
├── main.py
├── hmm.py
├── plot_results.py
├── recording_audio.py
├── setup_dataset.py
├── requirements.txt
├── likelihood_plot.png
├── prediction_distribution.png
├── report.pdf

---

## Requirements

Python 3.8 or higher is required.

Install dependencies:

pip install -r requirements.txt

Dependencies include:
- numpy
- scipy
- librosa
- scikit-learn
- matplotlib
- pyaudio
- sounddevice
- soundfile

---

## Dataset

The dataset consists of 40 self-recorded audio samples (10 per class).

Due to size constraints, audio files are not included in the repository. You can record your own samples using:

python recording_audio.py

Audio format:
- mono WAV
- 16 kHz
- 16-bit
- 3–5 seconds

Organize data as:

audio/
  happy/
  sad/
  angry/
  neutral/

---

## How to Run

Step 1 — Install dependencies:
pip install -r requirements.txt

Step 2 — Train HMMs and run evaluation:
python main.py --input-folder audio

Step 3 — Generate plots:
python plot_results.py

Outputs:
- likelihood_plot.png
- prediction_distribution.png

---

## Results

Emotion | Predicted | % | Avg Log-Likelihood
--------|----------|----|-------------------
Sad     | 23       | 57.5 | -3700
Happy   | 8        | 20.0 | -3800
Angry   | 6        | 15.0 | -3800
Neutral | 3        | 7.5  | -4000

Overall accuracy: ~27.5% (chance baseline: 25%)

---

## Key Implementation Parameters

- Feature dimension: 41
- HMM states per class: 5 (left-to-right)
- Emission model: single full-covariance Gaussian per state
- Training algorithm: Baum-Welch (EM)
- Max iterations: 100
- Convergence threshold: 1e-4
- Initialization: k-means on training frames
- Evaluation: Leave-one-out cross-validation (LOOCV)

---

## Reproducibility

To reproduce results:
1. Install dependencies
2. Provide audio data
3. Run training
4. Generate plots

Results may vary slightly due to random initialization.

---

## Author

Pranav Gupta  
CS5100 Foundations of Artificial Intelligence  
Spring 2026
