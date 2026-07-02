"""
trigger_word_detect.py
==================================================================
Companion script for Chapter 9 of "Deep Learning for Beginners:
Sequence Models" (AI Foundations Series, Book 5).

WHAT THIS SCRIPT DOES
-----------------------------------------------------------------
It builds and trains a TRIGGER-WORD DETECTOR - the kind of model
that listens to a stream of audio and fires the instant it hears a
wake word like "Alexa", "Hey Siri", or "OK Google". Our wake word
is the word "activate".

The model reads a SPECTROGRAM (a picture of sound: time across the
bottom, frequency up the side, brightness = energy) and outputs,
for each moment in time, a number between 0 and 1: "did the trigger
word JUST finish being spoken here?" A spike toward 1 means yes.

This mirrors the Course-5 Week-3 "Trigger Word Detection" lab. The
real lab uses actual .wav recordings; to stay self-contained and
instant, this script SYNTHESIZES spectrogram-shaped data with the
same structure the real pipeline produces:

    - a noisy "background"
    - a few "activate" bursts laid on top at random times
    - some "negative" word bursts that must NOT trigger it
    - labels that turn ON (1) for a short window right AFTER each
      "activate" burst finishes

The model architecture is exactly the lab's: a 1-D convolution to
summarize the spectrogram, two GRU layers to read it over time, and
a per-time-step sigmoid that emits the 0/1 trigger signal.
==================================================================
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # quiet TensorFlow start-up logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"      # deterministic math ordering

import numpy as np

SEED = 3
np.random.seed(SEED)

# -----------------------------------------------------------------
# Dimensions (same meaning as the lab)
# -----------------------------------------------------------------
Tx = 1101          # input time steps of the spectrogram (lab uses 5511; smaller = faster)
n_freq = 101       # frequency bins per time step (the height of the spectrogram)
# The Conv1D (kernel 15, stride 4) shrinks the time axis. Ty is exactly the
# number of columns it produces:  floor((Tx - kernel) / stride) + 1.
_KERNEL, _STRIDE = 15, 4
Ty = (Tx - _KERNEL) // _STRIDE + 1               # = 272 for Tx = 1101

# The word bursts occupy this many spectrogram columns.
POS_LEN = 40       # an "activate" burst
NEG_LEN = 30       # a different word (must NOT trigger)


# =================================================================
# 1. Synthesize spectrogram-shaped training data
# =================================================================
# A real spectrogram comes from an FFT of the waveform. We fake the
# SHAPE of one: a (Tx, n_freq) grid of energy. Background is low noise;
# a "word" is a brighter blob with a characteristic frequency pattern.

def make_word_pattern(length, base_freq, seed):
    """A little (length, n_freq) blob standing in for a spoken word."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, length)[:, None]
    f = np.linspace(0, 1, n_freq)[None, :]
    # energy concentrated around a few frequency bands that drift over time
    pattern = np.exp(-((f - base_freq - 0.1 * np.sin(4 * t)) ** 2) / 0.01)
    pattern += 0.4 * np.exp(-((f - base_freq / 2) ** 2) / 0.02)
    pattern *= (0.6 + 0.4 * rng.random((length, n_freq)))   # texture
    return pattern


def insert_ones(y, segment_end_t):
    """Turn on the label for 50 output steps right AFTER a trigger ends.

    This is the key labeling trick from the lab. We do NOT label the exact
    instant the word ends; we label a short WINDOW just after it, so the
    model learns to fire "the trigger just happened". Labeling a window
    (not a single step) also eases the huge imbalance between 0s and 1s.
    """
    end_out = int(segment_end_t * Ty / Tx)
    for i in range(end_out + 1, end_out + 51):
        if i < Ty:
            y[i] = 1.0
    return y


def make_example(seed):
    """Build one (spectrogram, label) training example."""
    rng = np.random.default_rng(seed)
    X = 0.1 * rng.random((Tx, n_freq))            # quiet background noise
    y = np.zeros(Ty)

    used = []
    def free_slot(length):
        for _ in range(20):
            start = rng.integers(0, Tx - length)
            if all(not (start < u_end and u_start < start + length)
                   for u_start, u_end in used):
                used.append((start, start + length))
                return start
        return None

    # Insert 1-3 "activate" bursts (positive), each followed by a label window.
    for _ in range(rng.integers(1, 4)):
        start = free_slot(POS_LEN)
        if start is None:
            continue
        X[start:start + POS_LEN] += make_word_pattern(POS_LEN, 0.6, rng.integers(1e9))
        insert_ones(y, start + POS_LEN)           # label the window AFTER it

    # Insert 0-2 "negative" word bursts (must NOT trigger -> no label).
    for _ in range(rng.integers(0, 3)):
        start = free_slot(NEG_LEN)
        if start is None:
            continue
        X[start:start + NEG_LEN] += make_word_pattern(NEG_LEN, 0.25, rng.integers(1e9))

    return X.astype("float32"), y.astype("float32")


def make_dataset(n, seed0=0):
    """Stack n examples into arrays the model can train on."""
    Xs, Ys = [], []
    for k in range(n):
        X, y = make_example(seed0 + k)
        Xs.append(X)
        Ys.append(y[:, None])                     # (Ty, 1)
    return np.array(Xs), np.array(Ys)


# =================================================================
# 2. The model (identical structure to the lab)
# =================================================================
def build_model(input_shape):
    """Conv1D -> BatchNorm/ReLU/Dropout -> 2x GRU -> per-step sigmoid.

    - The Conv1D slides over time and turns raw spectrogram columns into
      196 higher-level features. Its stride of 4 also shrinks the time
      axis (that is why Ty < Tx).
    - Two GRU layers read those features in order, building up context so
      the model can recognize the shape of the whole trigger word.
    - TimeDistributed(Dense(1, sigmoid)) makes ONE 0-to-1 prediction at
      EVERY output time step: "did the trigger just finish here?"
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv1D, BatchNormalization, Activation, Dropout,
        GRU, TimeDistributed, Dense)

    X_input = Input(shape=input_shape)

    X = Conv1D(196, kernel_size=15, strides=4)(X_input)   # summarize + shrink time
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Dropout(0.8)(X)

    X = GRU(128, return_sequences=True)(X)                # read over time
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)

    X = GRU(128, return_sequences=True)(X)                # read again, deeper
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)

    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # 0/1 at every step
    return Model(inputs=X_input, outputs=X)


# =================================================================
# 3. Plot the time-aligned output
# =================================================================
def plot_prediction(model, X, y_true, filename="trigger_prediction.png"):
    """Show the spectrogram, the true label, and the model's prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = model.predict(X[None, :, :], verbose=0)[0, :, 0]   # (Ty,)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6),
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.imshow(X.T, aspect="auto", origin="lower", cmap="viridis")
    ax1.set_ylabel("frequency bin")
    ax1.set_title("Input spectrogram  (bright blobs = spoken words)")

    tt = np.linspace(0, Tx, Ty)
    ax2.plot(tt, y_true[:, 0], label="true label", color="#548235", lw=2)
    ax2.plot(tt, pred, label="model output", color="#2E75B6", lw=2)
    ax2.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("time  ->")
    ax2.set_ylabel("trigger?")
    ax2.set_title('Output spikes toward 1 right AFTER each "activate"')
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    print(f"   saved {filename}")


# =================================================================
# MAIN
# =================================================================
def main():
    try:
        import tensorflow as tf
    except Exception as exc:                                # noqa: BLE001
        print("This script needs TensorFlow.  pip install tensorflow")
        print(f"({exc})")
        return
    tf.random.set_seed(SEED)

    print("Synthesizing spectrogram training data ...")
    X, Y = make_dataset(200, seed0=0)
    X_dev, Y_dev = make_dataset(40, seed0=10000)
    print(f"   train {X.shape}, labels {Y.shape}")

    model = build_model((Tx, n_freq))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    print("\nModel structure:")
    model.summary(line_length=70)

    print("\nTraining ...")
    model.fit(X, Y, batch_size=16, epochs=15, verbose=2)

    loss, acc = model.evaluate(X_dev, Y_dev, verbose=0)
    print(f"\nDev accuracy: {acc:.3f}  (fraction of time steps labeled correctly)")

    print("\nDrawing a prediction for one example ...")
    plot_prediction(model, X_dev[0], Y_dev[0])
    print("\nDone. Open trigger_prediction.png: the blue output should spike")
    print('toward 1 shortly after each "activate" blob in the spectrogram,')
    print("and stay near 0 for background noise and other words.")


if __name__ == "__main__":
    main()
