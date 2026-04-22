import os
import numpy as np
import librosa
from scipy.signal import medfilt

# ==============================
# HMM CLASS (TRUE HMM)
# ==============================
class HMM:

    def __init__(self, n_states):
        self.n_states = n_states
        self.random_state = np.random.RandomState(0)

        self.prior = self._normalize(self.random_state.rand(n_states))
        self.A = self._stochasticize(self.random_state.rand(n_states, n_states))

        self.mu = None
        self.var = None
        self.n_dims = None

    def _normalize(self, x):
        return (x + 1e-8) / np.sum(x + 1e-8)

    def _stochasticize(self, x):
        return (x + 1e-8) / np.sum(x + 1e-8, axis=1, keepdims=True)

    # ==============================
    # DIAGONAL GAUSSIAN
    # ==============================
    def _emission_log_prob(self, obs):
        T = obs.shape[1]
        logB = np.zeros((self.n_states, T))

        for s in range(self.n_states):
            diff = obs - self.mu[:, s].reshape(-1, 1)
            var = self.var[:, s].reshape(-1, 1)

            log_prob = -0.5 * (
                np.sum(np.log(2 * np.pi * var)) +
                np.sum((diff ** 2) / var, axis=0)
            )

            logB[s] = log_prob

        return logB

    # ==============================
    # FORWARD (LOG SPACE)
    # ==============================
    def _forward(self, logB):
        T = logB.shape[1]
        log_alpha = np.zeros((self.n_states, T))

        log_alpha[:, 0] = np.log(self.prior + 1e-10) + logB[:, 0]

        for t in range(1, T):
            temp = log_alpha[:, t-1].reshape(-1,1) + np.log(self.A + 1e-10)
            log_alpha[:, t] = logB[:, t] + np.logaddexp.reduce(temp, axis=0)

        return log_alpha

    # ==============================
    # BACKWARD (LOG SPACE)
    # ==============================
    def _backward(self, logB):
        T = logB.shape[1]
        log_beta = np.zeros((self.n_states, T))

        for t in range(T-2, -1, -1):
            temp = np.log(self.A + 1e-10) + logB[:, t+1] + log_beta[:, t+1]
            log_beta[:, t] = np.logaddexp.reduce(temp, axis=1)

        return log_beta

    # ==============================
    # TRAIN (BAUM-WELCH)
    # ==============================
    def fit(self, sequences, n_iter=10):

        self.n_dims = sequences[0].shape[0]

        # Initialize
        self.mu = sequences[0][:, self.random_state.choice(sequences[0].shape[1], self.n_states)]
        self.var = np.ones((self.n_dims, self.n_states))

        for _ in range(n_iter):

            A_num = np.zeros_like(self.A)
            A_den = np.zeros(self.n_states)

            mu_num = np.zeros_like(self.mu)
            var_num = np.zeros_like(self.var)
            gamma_sum = np.zeros(self.n_states)

            for obs in sequences:

                logB = self._emission_log_prob(obs)
                log_alpha = self._forward(logB)
                log_beta = self._backward(logB)

                log_gamma = log_alpha + log_beta
                log_gamma -= np.logaddexp.reduce(log_gamma, axis=0)

                gamma = np.exp(log_gamma)

                # ===== xi for transitions =====
                for t in range(obs.shape[1] - 1):
                    temp = (
                        log_alpha[:, t].reshape(-1,1)
                        + np.log(self.A + 1e-10)
                        + logB[:, t+1]
                        + log_beta[:, t+1]
                    )

                    temp -= np.logaddexp.reduce(temp.flatten())
                    xi = np.exp(temp)

                    A_num += xi
                    A_den += np.sum(gamma[:, t])

                # ===== emission updates =====
                for s in range(self.n_states):
                    weight = gamma[s]
                    gamma_sum[s] += np.sum(weight)

                    mu_num[:, s] += np.sum(obs * weight, axis=1)

                    diff = obs - self.mu[:, s].reshape(-1,1)
                    var_num[:, s] += np.sum((diff**2) * weight, axis=1)

            # update A
            self.A = self._stochasticize(A_num)

            # update means/variances
            for s in range(self.n_states):
                if gamma_sum[s] > 0:
                    self.mu[:, s] = mu_num[:, s] / gamma_sum[s]
                    self.var[:, s] = var_num[:, s] / gamma_sum[s] + 1e-6

        return self

    # ==============================
    # SCORE
    # ==============================
    def score(self, obs):
        logB = self._emission_log_prob(obs)
        log_alpha = self._forward(logB)
        return np.logaddexp.reduce(log_alpha[:, -1])


# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    energy = librosa.feature.rms(y=y)

    try:
        pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    except:
        pitch = np.zeros(mfcc.shape[1])

    pitch = np.nan_to_num(pitch)

    min_len = min(mfcc.shape[1], len(pitch), energy.shape[1])

    mfcc = mfcc[:, :min_len]
    delta = delta[:, :min_len]
    delta2 = delta2[:, :min_len]
    pitch = pitch[:min_len]
    energy = energy[:, :min_len]

    energy_flat = energy.flatten()
    threshold = np.percentile(energy_flat, 60)

    pitch[energy_flat <= threshold] = 0
    pitch = medfilt(pitch, kernel_size=5)
    pitch = pitch.reshape(1, -1)

    features = np.vstack([
        mfcc,
        delta,
        delta2,
        0.3 * pitch,
        0.3 * energy
    ])

    mask = energy_flat > threshold
    features = features[:, mask]

    return features


# ==============================
# TRAIN MODELS
# ==============================
def train_models(dataset_path):

    models = {}

    for emotion in os.listdir(dataset_path):
        path = os.path.join(dataset_path, emotion)
        if not os.path.isdir(path):
            continue

        print("Training:", emotion)

        sequences = []
        for file in os.listdir(path):
            if file.endswith(".wav"):
                f = extract_features(os.path.join(path, file))
                if f.shape[1] > 0:
                    sequences.append(f)

        if len(sequences) == 0:
            continue

        model = HMM(n_states=8)
        model.fit(sequences)

        models[emotion] = model

    return models


# ==============================
# PREDICT
# ==============================
def predict(models, file_path):

    features = extract_features(file_path)
    scores = {}

    for label, model in models.items():
        try:
            score = model.score(features)
        except:
            score = -np.inf

        scores[label] = score

    return max(scores, key=scores.get), scores


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    dataset_path = "dataset"
    test_file = "test.wav"

    models = train_models(dataset_path)
    pred, scores = predict(models, test_file)

    print("\nPrediction:", pred)
    print("Scores:", scores)