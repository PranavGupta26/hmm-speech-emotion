# HMM Speech Emotion Recognition

Reproduction of an HMM-based speech emotion recognition system with MFCC + pitch + energy features.

## Overview

This project implements a Hidden Markov Model classifier for four emotion classes:
**happy, sad, angry, neutral**.

## Features
- MFCC, delta, delta-delta
- Pitch and energy features
- Gaussian emission HMM
- Real-time prediction

## How to Run

pip install -r requirements.txt  
python3 main.py --input-folder audio  

## Dataset
Self-recorded dataset (~40 samples). Not included due to size.

## Outputs
- likelihood_plot.png
- prediction_distribution.png

## Results

| Emotion | Count | % | Log-Likelihood |
|--------|------|----|---------------|
| Sad | 23 | 57.5 | -3700 |
| Happy | 8 | 20.0 | -3800 |
| Angry | 6 | 15.0 | -3800 |
| Neutral | 3 | 7.5 | -4000 |

## Code Structure
- main.py → run system
- hmm.py → HMM model
- plot_results.py → graphs

## Author
Pranav Gupta
