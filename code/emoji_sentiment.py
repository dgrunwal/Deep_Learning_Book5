"""
emoji_sentiment.py
==================================================================
Companion script for Chapter 6 of "Deep Learning for Beginners:
Sequence Models" (AI Foundations Series, Book 5).

WHAT THIS SCRIPT DOES
-----------------------------------------------------------------
It predicts an emoji for a short sentence. For example:

    "I love you"          -> ❤️
    "lets play ball"      -> ⚾
    "food is ready"       -> 🍴
    "I am so happy"       -> 😄
    "this is hopeless"    -> 😞

It does this TWO different ways, so you can compare them:

  MODEL 1 - "Average then classify" (the simple baseline)
      Turn every word into its pretrained GloVe vector, AVERAGE
      those vectors into one sentence vector, and feed that single
      vector into a tiny softmax classifier. No word order at all.
      Implemented from scratch in NumPy so every step is visible.

  MODEL 2 - "Embeddings + LSTM" (the order-aware model)
      Feed the word vectors, in order, into an LSTM. The LSTM
      reads the sentence left to right and can therefore tell
      "not happy" from "happy". Built with Keras.

WHY BOTH?
      Averaging is a great first try and often works. But it is
      blind to order: "not happy" and "happy not" average to the
      same vector. The LSTM fixes that by reading words in sequence.

This mirrors the two halves of the Course-5 "Emojify" lab.
==================================================================
"""

# --- Keep TensorFlow quiet and reproducible. ---
# These MUST be set BEFORE tensorflow is imported anywhere, so they go
# first, before any other import. They only silence start-up chatter and
# fix numeric ordering - they do not change what the model learns.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # hide TF INFO/WARNING startup logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # deterministic math ordering
os.environ["PYTHONHASHSEED"] = "0"          # stable hashing across runs

import numpy as np

# One seed used everywhere, so every run gives the SAME result. Change it
# (or comment the seeding out) if you want to see the natural run-to-run
# variation a tiny dataset produces.
SEED = 3
np.random.seed(SEED)

# -----------------------------------------------------------------
# 0. A note on the GloVe vectors
# -----------------------------------------------------------------
# GloVe ("Global Vectors") is a set of PRETRAINED word vectors: a big
# lookup table where each English word already has a 50-number vector
# that captures its meaning. We did not train these; we download them.
# Words with similar meanings have similar vectors. That is the whole
# reason a classifier can learn from only a few hundred examples - the
# "meaning" work was already done for us on billions of words of text.
#
# A real run loads them from a file:
#     word_to_index, index_to_word, word_to_vec_map = \
#         read_glove_vecs('data/glove.6B.50d.txt')
#
# So that this file runs anywhere with no downloads, we build a tiny
# fake "GloVe" below. The CODE is identical to the real thing; only
# the size of the vocabulary changes.


def build_toy_glove(dim=50, seed=1):
    """Return (word_to_vec_map, word_to_index, index_to_word).

    In real use these three come from read_glove_vecs() over the real
    400,000-word GloVe file. Here we fake a handful of words so the
    script is self-contained and instant to run.
    """
    rng = np.random.default_rng(seed)
    vocab = [
        "i", "you", "love", "treasure", "adore", "happy", "joy", "glad",
        "sad", "unhappy", "miserable", "hopeless", "not",
        "food", "eat", "lunch", "ready", "hungry",
        "play", "ball", "game", "baseball", "fun", "lets",
        "am", "so", "this", "is", "feeling",
    ]
    # Give related words deliberately similar vectors so the toy demo
    # actually learns something. (Real GloVe gets this from data.)
    seeds = {
        "love": rng.normal(0, 1, dim), "happy": rng.normal(0, 1, dim),
        "sad": rng.normal(0, 1, dim), "food": rng.normal(0, 1, dim),
        "play": rng.normal(0, 1, dim),
    }
    family = {
        "love": ["treasure", "adore"], "happy": ["joy", "glad", "fun", "feeling"],
        "sad": ["unhappy", "miserable", "hopeless"],
        "food": ["eat", "lunch", "ready", "hungry"],
        "play": ["ball", "game", "baseball", "lets"],
    }
    word_to_vec_map = {}
    for w in vocab:
        word_to_vec_map[w] = rng.normal(0, 1, dim) * 0.3
    for base, members in family.items():
        word_to_vec_map[base] = seeds[base]
        for m in members:
            word_to_vec_map[m] = seeds[base] + rng.normal(0, 0.25, dim)

    word_to_index = {w: i + 1 for i, w in enumerate(word_to_vec_map)}  # +1: index 0 reserved for padding
    index_to_word = {i: w for w, i in word_to_index.items()}
    return word_to_vec_map, word_to_index, index_to_word


# Five emoji classes, matching the Emojify lab.
EMOJI = {0: "\u2764\ufe0f", 1: "\u26be", 2: "\U0001F604", 3: "\U0001F61E", 4: "\U0001F374"}
#         0 = love (heart)  1 = baseball  2 = smile  3 = disappointed  4 = fork & knife


def softmax(z):
    """Turn a vector of scores into probabilities that sum to 1."""
    z = z - np.max(z)            # subtract max for numerical safety
    e = np.exp(z)
    return e / np.sum(e)


# =================================================================
# MODEL 1 - AVERAGE THEN CLASSIFY  (pure NumPy, no framework)
# =================================================================

def sentence_to_avg(sentence, word_to_vec_map):
    """Average the GloVe vectors of every word in a sentence.

    Step 1: lower-case the sentence and split it into words.
    Step 2: look up each word's vector and add them up.
    Step 3: divide by the number of words found.

    The result is ONE vector that summarizes the whole sentence -
    but it has thrown away word order completely.
    """
    any_word = next(iter(word_to_vec_map))
    words = sentence.lower().split()
    avg = np.zeros(word_to_vec_map[any_word].shape)
    count = 0
    for w in words:
        if w in word_to_vec_map:          # skip words we have no vector for
            avg += word_to_vec_map[w]
            count += 1
    if count > 0:
        avg = avg / count
    return avg


def model_average(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    """Train a one-layer softmax classifier on averaged sentence vectors.

    This is logistic-regression-style training done by hand so you can
    see forward propagation, the cross-entropy cost, the gradients, and
    the weight update all in one place.

    X -- array of sentence strings, shape (m,)
    Y -- integer labels 0..4, shape (m,)
    """
    np.random.seed(1)
    any_word = next(iter(word_to_vec_map))
    m = Y.shape[0]                                   # number of examples
    n_y = 5                                          # number of emoji classes
    n_h = word_to_vec_map[any_word].shape[0]         # vector size (50)

    # Xavier-style initialization keeps the starting scores sensible.
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    # One-hot version of the labels: class 3 -> [0,0,0,1,0]
    Y_oh = np.eye(n_y)[Y.reshape(-1)]

    for t in range(num_iterations):
        cost = 0.0
        for i in range(m):                           # stochastic: one example at a time
            avg = sentence_to_avg(X[i], word_to_vec_map)   # (n_h,)
            z = np.dot(W, avg) + b                          # forward: raw scores
            a = softmax(z)                                  # probabilities
            cost += -np.sum(Y_oh[i] * np.log(a + 1e-12))    # cross-entropy

            # Gradients of cross-entropy + softmax simplify to (a - y).
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz

            # Gradient-descent step.
            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 100 == 0:
            print(f"  [average model] epoch {t:4d}  cost = {cost:.4f}")
    return W, b


def predict_average(sentence, W, b, word_to_vec_map):
    """Predict an emoji class for one sentence with the average model."""
    avg = sentence_to_avg(sentence, word_to_vec_map)
    return int(np.argmax(softmax(np.dot(W, avg) + b)))


# =================================================================
# MODEL 2 - EMBEDDINGS + LSTM  (Keras; order-aware)
# =================================================================
# This half needs TensorFlow/Keras. It is wrapped in a function and a
# try/except so the NumPy half always runs even without TensorFlow.

def sentences_to_indices(X, word_to_index, max_len):
    """Convert sentences to padded rows of word indices for an Embedding layer.

    "i love you" with max_len=5 might become [12, 3, 2, 0, 0]
    where 0 is the padding index that fills out short sentences.
    The Embedding layer will turn each index into its GloVe vector.
    """
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        for j, w in enumerate(X[i].lower().split()):
            if w in word_to_index and j < max_len:
                X_indices[i, j] = word_to_index[w]
    return X_indices


def build_lstm_model(input_shape, word_to_vec_map, word_to_index):
    """Build: Input -> (pretrained, frozen) Embedding -> LSTM -> LSTM -> softmax.

    The embedding layer is loaded with GloVe vectors and FROZEN
    (trainable=False) so the small dataset cannot wreck them. Only the
    LSTM and the final Dense layer learn.
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Dense, Activation, Dropout, LSTM, Embedding)

    # --- Build the pretrained embedding layer ---
    vocab_len = len(word_to_index) + 1               # +1 for the padding row
    any_word = next(iter(word_to_vec_map))
    emb_dim = word_to_vec_map[any_word].shape[0]
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]   # row idx = that word's vector

    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])        # load GloVe, then freeze

    # --- Wire up the network ---
    sentence_indices = Input(shape=input_shape, dtype="int32")
    embeddings = embedding_layer(sentence_indices)   # indices -> vectors
    X = LSTM(128, return_sequences=True)(embeddings)  # first LSTM returns a sequence
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)          # second LSTM returns one vector
    X = Dropout(0.5)(X)
    X = Dense(5)(X)                                   # 5 emoji scores
    X = Activation("softmax")(X)
    return Model(inputs=sentence_indices, outputs=X)


def run_lstm_demo(X_train, Y_train, word_to_vec_map, word_to_index, max_len):
    """Train and show the LSTM model, if TensorFlow is available."""
    try:
        import tensorflow as tf
        from tensorflow.keras.utils import to_categorical
    except Exception as exc:                          # noqa: BLE001
        print("\n[LSTM model] TensorFlow not installed - skipping this half.")
        print(f"             ({exc})")
        print("             Install with:  pip install tensorflow")
        return

    # Seed TensorFlow too, so the LSTM trains the same way every run.
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)   # seeds Python, NumPy, and TF together

    print("\n[LSTM model] Building embeddings + LSTM network ...")
    model = build_lstm_model((max_len,), word_to_vec_map, word_to_index)
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    X_idx = sentences_to_indices(X_train, word_to_index, max_len)
    Y_oh = to_categorical(Y_train, num_classes=5)
    model.fit(X_idx, Y_oh, epochs=60, batch_size=8, shuffle=True, verbose=0)

    # Test sentences. The "happy" / "not happy" pair is the key contrast:
    # the LSTM should now give them DIFFERENT emojis, because we taught it
    # what "not" does. The averaging model can never tell them apart.
    print("\nPredictions from the LSTM model:")
    for s in ["i adore you", "lets play ball", "i am hungry",
              "happy", "not happy", "sad", "not sad"]:
        idx = sentences_to_indices(np.array([s]), word_to_index, max_len)
        pred = int(np.argmax(model.predict(idx, verbose=0)))
        print(f"   {s:20s} -> {EMOJI[pred]}")

    print("\n   Notice: 'happy' and 'not happy' now get DIFFERENT emojis.")
    print("   The LSTM read the word order and let 'not' flip the meaning -")
    print("   something the averaging model cannot do, because to it both")
    print("   sentences are just the same bag of words.")


# =================================================================
# MAIN
# =================================================================

def main():
    word_to_vec_map, word_to_index, index_to_word = build_toy_glove()

    # The BASE training set, used by BOTH models (sentence, emoji-class).
    X_base = np.array([
        "i love you", "i adore you", "i treasure you",
        "lets play ball", "lets play baseball", "this is a fun game",
        "food is ready", "i am hungry", "lunch is ready",
        "i am happy", "this is so glad", "i feeling joy",
        "this is hopeless", "i am miserable", "i am unhappy",
    ])
    Y_base = np.array([0, 0, 0, 1, 1, 1, 4, 4, 4, 2, 2, 2, 3, 3, 3])

    # EXTRA negation examples - sentences where "not" flips the feeling.
    # We give these ONLY to the LSTM. The point: the LSTM has the MACHINERY
    # to read word order, but it still has to be SHOWN what "not" does. A
    # model only learns patterns it has actually seen in its training data.
    # (We deliberately keep them OUT of the averaging model, because
    # averaging can never use word order no matter what it is trained on.)
    X_neg = np.array([
        "not happy", "i am not happy", "i am not glad",      # -> sad
        "not sad", "i am not sad", "i am not unhappy",       # -> happy
    ])
    Y_neg = np.array([3, 3, 3, 2, 2, 2])

    print("=" * 60)
    print("MODEL 1: average the word vectors, then classify (NumPy)")
    print("=" * 60)
    W, b = model_average(X_base, Y_base, word_to_vec_map)

    print("\nPredictions from the average model:")
    for s in ["i love you", "lets play baseball", "food is ready",
              "i am happy", "this is hopeless"]:
        pred = predict_average(s, W, b, word_to_vec_map)
        print(f"   {s:20s} -> {EMOJI[pred]}")

    print("\nThe weakness of averaging - word order is invisible.")
    print("These two contain the same words, so averaging scores them alike:")
    for s in ["glad", "not glad"]:
        pred = predict_average(s, W, b, word_to_vec_map)
        print(f"   {s:20s} -> {EMOJI[pred]}   (averaging cannot use the 'not')")

    print("\n" + "=" * 60)
    print("MODEL 2: embeddings + LSTM, which CAN read word order (Keras)")
    print("=" * 60)
    # The LSTM trains on the base sentences PLUS the negation examples,
    # so it gets the chance to learn what "not" does.
    X_lstm = np.concatenate([X_base, X_neg])
    Y_lstm = np.concatenate([Y_base, Y_neg])
    run_lstm_demo(X_lstm, Y_lstm, word_to_vec_map, word_to_index, max_len=10)

    print("\n" + "=" * 60)
    print("WHAT THIS COMPARISON SHOWS")
    print("=" * 60)
    print("Averaging is simple and fast, but blind to word order: to it,")
    print("'happy' and 'not happy' are the same bag of words, so they get")
    print("the same emoji.")
    print()
    print("The LSTM reads words IN ORDER, so it CAN tell them apart - but")
    print("only because we gave it a few 'not ...' training examples to")
    print("learn from. The machinery to read order is not enough on its own;")
    print("a model only learns patterns it has actually seen in its data.")
    print()
    print("Think of the LSTM as a capable new car and the training data as")
    print("driving lessons: a great car still cannot take a turn it was")
    print("never taught. Our toy dataset is tiny, so if a result looks off,")
    print("that is the lesson - on the full Emojify dataset (hundreds of")
    print("real sentences and real GloVe vectors) the LSTM handles cases")
    print("like negation far better than averaging ever could.")


if __name__ == "__main__":
    main()
