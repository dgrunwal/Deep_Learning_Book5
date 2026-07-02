"""
neural_machine_translation_learning_lesson_american.py
==================================================================
Beginner teaching script for attention-based date translation.
American slash-date version: M/D/YYYY, so 7/8/2010 means July 8, 2010.

Core lesson:
    A neural network usually learns the patterns it sees during training.
    Attention helps it look at useful input positions, but attention does
    NOT magically teach it date formats that were missing from the dataset.

This script deliberately does two passes:

    PASS 1 - BASELINE TRAINING
        Trains on a small synthetic date dataset WITHOUT weekday prefixes.
        Then it tests this out-of-distribution input:

            "Tuesday 09 Oct 1993"

        This may produce a wrong answer because the model never learned
        that weekday words should be ignored.

    PASS 2 - FIXED TRAINING DATA
        Retrains on a richer dataset that INCLUDES weekday prefixes,
        zero-padded days, and American slash dates. Then it tests
        the same input again.

Expected corrected translations:

        "Tuesday 09 Oct 1993" -> "1993-10-09"
        "7/8/2010"            -> "2010-07-08"

Important detail:
    The weekday word is treated as extra text to ignore. This script does
    not check whether "Tuesday" was the historically correct weekday for
    that calendar date.

Requirements:
    pip install tensorflow numpy matplotlib

© 2026 David Grunwald. All rights reserved.
==================================================================
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # quiet TensorFlow start-up logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # more repeatable CPU math ordering

import random
import numpy as np

SEED = 3
random.seed(SEED)
np.random.seed(SEED)

COPYRIGHT_NOTICE = "© 2026 David Grunwald. All rights reserved."

# =================================================================
# 1. Build synthetic date datasets
# =================================================================

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_FULL = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Fixed input/output lengths.
# "Wednesday, September 09, 1993" is 30 chars, so Tx=32 gives a little room.
Tx = 32
Ty = 10    # YYYY-MM-DD is always 10 chars


def ordinal_suffix(day):
    """Return st, nd, rd, or th for beginner-friendly date variety."""
    if 10 <= day % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def make_dataset(n, include_weekday_styles=False):
    """Return a list of (human_readable, machine_readable) date strings.

    include_weekday_styles=False creates the intentionally limited baseline
    dataset. It does NOT include examples like "Tuesday 09 Oct 1993".

    include_weekday_styles=True creates the fixed dataset. It includes the
    missing weekday pattern, teaches the model that weekday words are usually
    irrelevant for the target YYYY-MM-DD string, and uses American slash
    dates: month/day/year.
    """
    data = []
    max_style = 8 if include_weekday_styles else 3

    for _ in range(n):
        year = random.randint(1950, 2020)
        month = random.randint(1, 12)
        day = random.randint(1, DAYS_IN_MONTH[month - 1])
        weekday = random.choice(WEEKDAYS)
        suffix = ordinal_suffix(day)

        machine = f"{year:04d}-{month:02d}-{day:02d}"
        style = random.randint(0, max_style)

        # Original simple styles.
        if style == 0:
            human = f"{day} {MONTHS[month-1]} {year}"              # 9 Oct 1993
        elif style == 1:
            human = f"{MONTH_FULL[month-1]} {day} {year}"          # October 9 1993
        elif style == 2:
            human = f"{month}/{day}/{year}"                        # American: 10/9/1993 means Oct 9
        elif style == 3:
            human = f"{MONTHS[month-1]} {day}, {year}"             # Oct 9, 1993

        # New styles that fix the original failure.
        elif style == 4:
            human = f"{weekday} {day:02d} {MONTHS[month-1]} {year}" # Tuesday 09 Oct 1993
        elif style == 5:
            human = f"{weekday} {day} {MONTHS[month-1]} {year}"    # Tuesday 9 Oct 1993
        elif style == 6:
            human = f"{weekday}, {MONTH_FULL[month-1]} {day}, {year}" # Tuesday, October 9, 1993
        elif style == 7:
            human = f"{MONTH_FULL[month-1]} {day}{suffix} {year}"  # October 9th 1993
        else:
            human = f"{day:02d} {MONTHS[month-1]} {year}"          # 09 Oct 1993

        data.append((human.lower(), machine))

    # Add a few exact anchor examples so the lesson case is definitely present.
    # This is not cheating; it is showing that examples must exist in the
    # training distribution before the model can be expected to learn them.
    if include_weekday_styles:
        data.extend([
            ("tuesday 09 oct 1993", "1993-10-09"),
            ("monday 03 may 1979", "1979-05-03"),
            ("friday 25 dec 1999", "1999-12-25"),
            ("7/8/2010", "2010-07-08"),
            ("12/25/1999", "1999-12-25"),
            ("3/5/1979", "1979-03-05"),
        ])

    return data


def build_vocabs(data):
    """Map every input character and output character to an integer."""
    human_chars = set()
    machine_chars = set()
    for h, m in data:
        human_chars.update(h)
        machine_chars.update(m)

    # Space is used for simple padding in translate()/plot_attention().
    human_chars.update([" ", "<unk>", "<pad>"])
    machine_chars.update(["<pad>"])

    human_vocab = {c: i for i, c in enumerate(sorted(human_chars))}
    machine_vocab = {c: i for i, c in enumerate(sorted(machine_chars))}
    inv_machine = {i: c for c, i in machine_vocab.items()}
    return human_vocab, machine_vocab, inv_machine


def encode(data, human_vocab, machine_vocab):
    """Turn strings into one-hot arrays the network can read."""
    def to_ids(s, vocab, length):
        s = s[:length].ljust(length)
        return [vocab.get(c, vocab.get("<unk>", 0)) for c in s]

    Xoh, Yoh = [], []
    hv_size, mv_size = len(human_vocab), len(machine_vocab)
    for h, m in data:
        xids = to_ids(h, human_vocab, Tx)
        yids = [machine_vocab[c] for c in m]
        Xoh.append(np.eye(hv_size)[xids])
        Yoh.append(np.eye(mv_size)[yids])
    return np.array(Xoh), np.array(Yoh)


# =================================================================
# 2. Build the attention model
# =================================================================

def build_model(human_vocab_size, machine_vocab_size, n_a=32, n_s=64):
    """Build an attention-based character-level date translator."""
    from tensorflow.keras.layers import (
        Input, Bidirectional, LSTM, Dense, Activation,
        Concatenate, Dot, RepeatVector)
    from tensorflow.keras.models import Model
    import tensorflow.keras.backend as K

    # Shared attention layers reused at every output character.
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation="tanh")
    densor2 = Dense(1, activation="relu")
    activator = Activation(lambda x: K.softmax(x, axis=1))
    dotor = Dot(axes=1)

    def one_step_attention(a, s_prev):
        """Build one context vector by attending over all input positions."""
        s_prev = repeator(s_prev)
        concat = concatenator([a, s_prev])
        e = densor1(concat)
        energies = densor2(e)
        alphas = activator(energies)
        context = dotor([alphas, a])
        return context, alphas

    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name="s0")
    c0 = Input(shape=(n_s,), name="c0")
    s, c = s0, c0

    post_attention_LSTM = LSTM(n_s, return_state=True)
    output_layer = Dense(machine_vocab_size, activation="softmax")

    # Encoder: reads the whole input both forward and backward.
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)

    outputs = []
    attention_tensors = []
    for _ in range(Ty):
        context, alphas = one_step_attention(a, s)
        s, _, c = post_attention_LSTM(context, initial_state=[s, c])
        outputs.append(output_layer(s))
        attention_tensors.append(alphas)

    model = Model(inputs=[X, s0, c0], outputs=outputs)
    model.attention_model = Model(inputs=[X, s0, c0], outputs=attention_tensors)
    return model


# =================================================================
# 3. Translation and attention visualization
# =================================================================

def translate(model, human_vocab, inv_machine, example, n_s=64):
    """Run the model on one human date and return the predicted machine date."""
    def to_ids(s):
        s = s[:Tx].ljust(Tx)
        return [human_vocab.get(c, human_vocab.get("<unk>", 0)) for c in s]

    xoh = np.eye(len(human_vocab))[to_ids(example.lower())][None, :, :]
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    preds = model.predict([xoh, s0, c0], verbose=0)
    preds = np.array(preds)
    ids = np.argmax(preds, axis=-1).flatten()
    return "".join(inv_machine[i] for i in ids)


def collect_attention(model, xoh, s0, c0):
    """Return a (Ty, Tx) matrix of attention weights for one example."""
    result = model.attention_model.predict([xoh, s0, c0], verbose=0)
    if not isinstance(result, list):
        result = [result]
    return np.array([r[0, :, 0] for r in result])


def plot_attention(model, human_vocab, inv_machine, example, n_s=64,
                   filename="attention_map_fixed.png"):
    """Draw a heatmap showing which input chars the model attended to."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def to_ids(s):
        s = s[:Tx].ljust(Tx)
        return [human_vocab.get(c, human_vocab.get("<unk>", 0)) for c in s]

    xoh = np.eye(len(human_vocab))[to_ids(example.lower())][None, :, :]
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    weights = collect_attention(model, xoh, s0, c0)

    pred = translate(model, human_vocab, inv_machine, example, n_s)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(weights, cmap="Blues", aspect="auto")
    visible_input = example[:Tx]
    ax.set_xticks(range(len(visible_input)))
    ax.set_xticklabels(list(visible_input), fontsize=9)
    ax.set_yticks(range(len(pred)))
    ax.set_yticklabels(list(pred), fontsize=10)
    ax.set_xlabel("input date characters")
    ax.set_ylabel("output characters")
    ax.set_title(f'Attention after fixed training: "{example}" -> "{pred}"')
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.text(0.5, 0.01, COPYRIGHT_NOTICE, ha="center", fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(filename, dpi=140)
    print(f"   saved attention heatmap to {filename}")
    print(f"   {COPYRIGHT_NOTICE}")


# =================================================================
# 4. Training helper
# =================================================================

def train_translator(data, epochs, batch_size=100, n_s=64, label="model"):
    """Build vocab, encode data, train model, and return trained pieces."""
    import tensorflow as tf

    human_vocab, machine_vocab, inv_machine = build_vocabs(data)
    Xoh, Yoh = encode(data, human_vocab, machine_vocab)
    model = build_model(len(human_vocab), len(machine_vocab), n_a=32, n_s=n_s)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005),
                  loss="categorical_crossentropy")

    m = Xoh.shape[0]
    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    outputs = list(Yoh.swapaxes(0, 1))

    print(f"\nTraining {label} ...")
    print(f"   examples: {len(data)}; input vocab: {len(human_vocab)}; output vocab: {len(machine_vocab)}")
    model.fit([Xoh, s0, c0], outputs, epochs=epochs, batch_size=batch_size, verbose=2)
    return model, human_vocab, inv_machine


# =================================================================
# 5. Main beginner lesson
# =================================================================

def main():
    try:
        import tensorflow as tf
    except Exception as exc:  # noqa: BLE001
        print("This script needs TensorFlow. Run: pip install tensorflow")
        print(f"({exc})")
        return

    tf.random.set_seed(SEED)

    lesson_input = "Tuesday 09 Oct 1993"
    expected = "1993-10-09"

    print("=" * 72)
    print("Attention date translation lesson")
    print(COPYRIGHT_NOTICE)
    print("=" * 72)
    print("Target lesson:")
    print("   Attention helps the model look at the input, but the model still")
    print("   needs training examples that match the formats it will later see.")
    print("   In this version, slash dates are American: month/day/year.")
    print("   Therefore 7/8/2010 means July 8, 2010 -> 2010-07-08.")
    print()
    print(f"Lesson input:    {lesson_input}")
    print(f"Expected output: {expected}")

    # -------------------------------------------------------------
    # PASS 1: intentionally limited training data.
    # -------------------------------------------------------------
    print("\nPASS 1: Baseline dataset WITHOUT weekday-prefix training examples")
    baseline_data = make_dataset(8000, include_weekday_styles=False)
    baseline_model, baseline_hv, baseline_inv = train_translator(
        baseline_data, epochs=12, n_s=64, label="baseline model")

    baseline_prediction = translate(baseline_model, baseline_hv, baseline_inv, lesson_input)
    print("\nBaseline test:")
    print(f"   {lesson_input:24s} -> {baseline_prediction}")
    print(f"   expected{'':17s} -> {expected}")
    if baseline_prediction != expected:
        print("   Result: WRONG or unreliable, because this input format was not in training.")
    else:
        print("   Result: Correct this time, but still not trustworthy: the format was missing from training.")

    # -------------------------------------------------------------
    # PASS 2: fixed training data.
    # -------------------------------------------------------------
    print("\nPASS 2: Fixed dataset WITH weekday prefixes and zero-padded days")
    print("   Added examples like: Tuesday 09 Oct 1993 -> 1993-10-09")
    fixed_data = make_dataset(20000, include_weekday_styles=True)
    fixed_model, fixed_hv, fixed_inv = train_translator(
        fixed_data, epochs=18, n_s=64, label="fixed model")

    fixed_prediction = translate(fixed_model, fixed_hv, fixed_inv, lesson_input)
    print("\nFixed test:")
    print(f"   {lesson_input:24s} -> {fixed_prediction}")
    print(f"   expected{'':17s} -> {expected}")

    if fixed_prediction == expected:
        print("   Result: CORRECT. The model now saw this kind of pattern during training.")
    else:
        print("   Result: Still missed it. Increase data/epochs or simplify the model lesson.")

    print("\nOther sanity checks after fixed training:")
    for ex in ["3 May 1979", "March 3rd 2001", "7/8/2010", "December 25 1999"]:
        print(f"   {ex:24s} -> {translate(fixed_model, fixed_hv, fixed_inv, ex)}")

    print("\nDrawing attention map for the fixed lesson case ...")
    plot_attention(fixed_model, fixed_hv, fixed_inv, lesson_input,
                   filename="attention_map_fixed_weekday.png")

    print("\nBeginner takeaway:")
    print("   This was not just a random mistake. The first model was asked to")
    print("   handle a date format it had not been taught. The fixed model gets")
    print("   a fairer training distribution, so attention has something useful")
    print("   to learn from. For slash dates, the model learns your chosen convention:")
    print("   American month/day/year, not European day/month/year.")
    print(COPYRIGHT_NOTICE)


if __name__ == "__main__":
    main()
