import os
import argparse
import numpy as np
import pyaudio
import wave
import librosa
import matplotlib.pyplot as plt
from collections import Counter

from hmm import train_models, predict  # your HMM file

# ==============================
# AUDIO SETTINGS
# ==============================

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "real_time_audio.wav"

# ==============================
# ARG PARSER
# ==============================

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Emotion Recognition using HMM')
    parser.add_argument("--input-folder", dest="input_folder", required=True,
                        help="Dataset folder containing emotion subfolders")
    return parser

# ==============================
# RECORD AUDIO
# ==============================

def record_audio():
    print("\nRecording...")

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return WAVE_OUTPUT_FILENAME

# ==============================
# GRAPH GENERATION
# ==============================

def generate_graphs(predictions, scores_list):

    print("\nGenerating graphs...")

    # ==============================
    # Prediction Distribution
    # ==============================
    count = Counter(predictions)

    plt.figure()
    plt.bar(count.keys(), count.values())
    plt.title("Prediction Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Frequency")
    plt.savefig("prediction_distribution.png")
    plt.close()

    # ==============================
    # Average Likelihood Plot
    # ==============================
    avg_scores = {}

    for key in scores_list[0].keys():
        avg_scores[key] = np.mean([s[key] for s in scores_list])

    labels = list(avg_scores.keys())
    values = list(avg_scores.values())

    plt.figure()
    plt.bar(labels, values)
    plt.title("Average Log-Likelihood per Class")
    plt.xlabel("Emotion")
    plt.ylabel("Log-Likelihood")
    plt.savefig("likelihood_plot.png")
    plt.close()

    print("Graphs saved as:")
    print("- prediction_distribution.png")
    print("- likelihood_plot.png")

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    dataset_path = args.input_folder

    # ==============================
    # TRAIN MODELS
    # ==============================
    print("\nTraining all emotion models...")
    models = train_models(dataset_path)
    print("\nTraining complete!")

    # ==============================
    # STORAGE FOR ANALYSIS
    # ==============================
    all_predictions = []
    all_scores = []

    try:
        # ==============================
        # REAL-TIME LOOP
        # ==============================
        while True:
            file_path = record_audio()

            predicted_emotion, scores = predict(models, file_path)

            print("\nPredicted Emotion:", predicted_emotion)
            print("Scores:", scores)
            print("\n-----------------------------")

            # Store results
            all_predictions.append(predicted_emotion)
            all_scores.append(scores)

    except KeyboardInterrupt:
        print("\n\nStopped recording.")

        if len(all_predictions) > 0:
            print("\nCollected", len(all_predictions), "samples.")

            generate_graphs(all_predictions, all_scores)
        else:
            print("No data collected.")