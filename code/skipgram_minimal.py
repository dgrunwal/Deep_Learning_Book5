"""
skipgram_minimal.py
===================================================================
A TINY skip-gram word-embedding model with negative sampling,
trained on a small corpus in pure NumPy. The whole objective is
visible -- you can read every line that makes the embeddings learn.

The one idea behind everything here
-------------------------------------------------------------------
"You shall know a word by the company it keeps." We learn a word's
meaning from the words that appear NEAR it. Skip-gram turns that
slogan into a guessing game:

    Given a CENTER word, can we tell which other words really sit
    near it (its neighbors) from random words that do not?

A word's vector is nudged, example by example, until words that keep
similar company end up with similar vectors. That is the entire
trick -- and it is how real tools like Word2Vec produced the
pretrained vectors you used in Chapter 4.

Why "negative sampling"?
-------------------------------------------------------------------
The original skip-gram asked the model to pick the right neighbor
out of the ENTIRE vocabulary with a softmax. For 10,000+ words that
sum is painfully slow. Negative sampling replaces that one giant
multiple-choice question with a handful of easy yes/no questions:

    (center, real neighbor)   -> answer YES   (label 1)
    (center, random word)     -> answer NO    (label 0)   x k times

Each question is simple logistic regression on the dot product of
two word vectors. No giant softmax -- just a few cheap updates per
step. That is what makes training tractable.

© 2026 David Grunwald. All rights reserved.
"""

import numpy as np
import re


# -------------------------------------------------------------------
# Tiny helpers
# -------------------------------------------------------------------
def sigmoid(z):
    """Squash a number to (0, 1): our 'probability they belong together'."""
    return 1.0 / (1.0 + np.exp(-z))


# -------------------------------------------------------------------
# 1. Build a vocabulary from a small corpus
# -------------------------------------------------------------------
def build_vocab(text):
    """Split text into words and map each unique word to an integer id."""
    words = re.findall(r"[a-z]+", text.lower())
    vocab = sorted(set(words))
    stoi = {w: i for i, w in enumerate(vocab)}      # word  -> id
    itos = {i: w for w, i in stoi.items()}          # id    -> word
    token_ids = [stoi[w] for w in words]            # the corpus as ids
    return token_ids, stoi, itos


# -------------------------------------------------------------------
# 2. Make (center, neighbor) training pairs from a sliding window
# -------------------------------------------------------------------
def make_pairs(token_ids, window=2):
    """For each word, pair it with every word within `window` slots.

    Sentence: a b c d   (window=1)
       center b -> neighbors a, c
       center c -> neighbors b, d
    These positive pairs are the 'words that really keep company'.
    """
    pairs = []
    for center_pos, center in enumerate(token_ids):
        lo = max(0, center_pos - window)
        hi = min(len(token_ids), center_pos + window + 1)
        for ctx_pos in range(lo, hi):
            if ctx_pos != center_pos:
                pairs.append((center, token_ids[ctx_pos]))
    return pairs


# -------------------------------------------------------------------
# 3. The model: two small embedding tables
# -------------------------------------------------------------------
class SkipGramNS:
    """Skip-gram with negative sampling.

    We keep TWO vectors per word:
      W_in  -- the vector used when the word is the CENTER word
      W_out -- the vector used when the word is a NEIGHBOR/context word
    Keeping them separate makes the math clean. After training we use
    W_in as 'the' embedding (a common choice).
    """

    def __init__(self, vocab_size, dim=10, seed=1):
        rng = np.random.RandomState(seed)
        # Small random starting vectors. Training will shape them.
        self.W_in = rng.randn(vocab_size, dim) * 0.1   # center vectors
        self.W_out = rng.randn(vocab_size, dim) * 0.1  # context vectors
        self.V = vocab_size

    def train_pair(self, center, context, negatives, lr):
        """One negative-sampling update for a single positive pair.

        We push the center vector and the TRUE context vector closer
        together (label 1), and push the center vector and each RANDOM
        negative vector apart (label 0). 'Closer/apart' is measured by
        the dot product fed through a sigmoid -- plain logistic
        regression, done once per word pair.
        """
        v_c = self.W_in[center]               # center word vector

        loss = 0.0
        # --- positive example: (center, real neighbor), target = 1 ---
        u_o = self.W_out[context]
        score = sigmoid(np.dot(v_c, u_o))     # predicted P(they belong)
        err = score - 1.0                     # gradient of logistic loss
        grad_c = err * u_o                    # accumulate update for v_c
        self.W_out[context] -= lr * err * v_c  # update the context vector
        loss -= np.log(score + 1e-9)

        # --- negative examples: (center, random word), target = 0 ---
        for neg in negatives:
            u_n = self.W_out[neg]
            score = sigmoid(np.dot(v_c, u_n))
            err = score - 0.0                 # target is 0 this time
            grad_c += err * u_n
            self.W_out[neg] -= lr * err * v_c
            loss -= np.log(1.0 - score + 1e-9)

        # apply the accumulated update to the center vector last
        self.W_in[center] -= lr * grad_c
        return loss


# -------------------------------------------------------------------
# 4. A 'noise' sampler for picking negative words
# -------------------------------------------------------------------
def make_neg_sampler(token_ids, vocab_size, seed=1):
    """Pick random words, but weighted by how common they are.

    Word2Vec raises each word's frequency to the 3/4 power, which
    slightly favors rarer words. We copy that recipe so the negatives
    feel realistic.
    """
    counts = np.bincount(token_ids, minlength=vocab_size).astype(np.float64)
    weights = counts ** 0.75
    weights /= weights.sum()
    rng = np.random.RandomState(seed)

    def sample(k, avoid):
        out = []
        while len(out) < k:
            w = rng.choice(vocab_size, p=weights)
            if w != avoid:          # don't sample the true context word
                out.append(w)
        return out

    return sample


# -------------------------------------------------------------------
# 5. Train and inspect
# -------------------------------------------------------------------
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def nearest(word, model, stoi, itos, k=3):
    if word not in stoi:
        return []
    v = model.W_in[stoi[word]]
    sims = [(cosine(v, model.W_in[i]), itos[i])
            for i in range(model.V) if itos[i] != word]
    sims.sort(reverse=True)
    return sims[:k]


def main():
    # A small corpus where some words clearly keep similar company.
    corpus = (
        "the cat sat on the mat . the dog sat on the rug . "
        "a cat chased a mouse . a dog chased a cat . "
        "the king ruled the land . the queen ruled the land . "
        "the king loved the queen . the man met the woman . "
        "a man walked the dog . a woman walked the cat . "
    ) * 60

    token_ids, stoi, itos = build_vocab(corpus)
    pairs = make_pairs(token_ids, window=2)
    sampler = make_neg_sampler(token_ids, len(stoi))
    model = SkipGramNS(vocab_size=len(stoi), dim=12)

    K = 5          # negative samples per positive pair
    lr = 0.025
    epochs = 8
    rng = np.random.RandomState(0)

    print(f"Vocabulary size : {len(stoi)} words")
    print(f"Training pairs  : {len(pairs)}")
    print(f"Negatives (K)   : {K}\n")
    print("Training skip-gram with negative sampling...\n")

    for epoch in range(epochs):
        rng.shuffle(pairs)
        total = 0.0
        for center, context in pairs:
            negs = sampler(K, avoid=context)
            total += model.train_pair(center, context, negs, lr)
        print(f"  epoch {epoch + 1}/{epochs}  |  avg loss {total / len(pairs):.4f}")

    print("\nNearest neighbors after training (by cosine similarity):")
    for word in ["cat", "king", "sat", "walked"]:
        nbrs = nearest(word, model, stoi, itos, k=3)
        shown = ", ".join(f"{w} ({s:+.2f})" for s, w in nbrs)
        print(f"  {word:>7}: {shown}")

    print("\nWords that keep similar company drift to similar vectors --")
    print("that is a word embedding, learned from nothing but context.")


if __name__ == "__main__":
    main()
