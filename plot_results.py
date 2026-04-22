import matplotlib.pyplot as plt
from collections import Counter

# ==============================
# PASTE YOUR DATA HERE
# ==============================

predictions = [
    "sad","neutral","sad","sad","sad","sad","neutral",
    "sad","happy","happy","sad","happy","sad","sad"
]

scores = {
    "happy": -4140,
    "sad": -3459,
    "neutral": -3874,
    "angry": -4098
}

# ==============================
# 1. Prediction Distribution
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
# 2. Likelihood Plot
# ==============================

labels = list(scores.keys())
values = list(scores.values())

plt.figure()
plt.bar(labels, values)
plt.title("Log-Likelihood Scores per Class")
plt.xlabel("Emotion")
plt.ylabel("Log-Likelihood")
plt.savefig("likelihood_plot.png")
plt.close()

print("Graphs generated successfully!")