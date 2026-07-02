"""
positional_encoding.py
==================================================================
Companion script for Chapter 10 of "Deep Learning for Beginners:
Sequence Models" (AI Foundations Series, Book 5).

WHAT THIS SCRIPT DOES
-----------------------------------------------------------------
The transformer looks at every word in a sentence at once, in
parallel. That speed comes at a price: unlike an RNN, which reads
words one after another, the transformer has NO built-in sense of
ORDER. To it, "dog bites man" and "man bites dog" are the same bag
of vectors. POSITIONAL ENCODING fixes this by adding a small,
position-dependent pattern to each word's vector, so position 0
looks a little different from position 1, and so on.

The trick (from "Attention Is All You Need") is to build those
patterns out of SINE and COSINE waves of many different
wavelengths. This script:

  1. Builds the sinusoidal positional-encoding matrix.
  2. Draws the classic "barcode" heatmap of it.
  3. Shows two sanity checks: nearby positions have SIMILAR
     encodings, and the similarity fades smoothly with distance.

It mirrors the Course-5 Week-4 positional-encoding lab, rewritten
in pure NumPy + matplotlib so it runs anywhere with no framework.

WHY SINE AND COSINE?
-----------------------------------------------------------------
Think of a row of clocks, each ticking at a different speed. The
fastest clock flips every step; the slowest barely moves over the
whole sentence. Reading all the clock hands at once gives every
position a unique "time stamp" - and, because waves are smooth,
positions that are close in the sentence get similar stamps. That
smoothness is what lets the model reason about "nearby" and "far".
==================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =================================================================
# 1. Build the positional-encoding matrix
# =================================================================
def get_angles(positions, d):
    """Compute the raw angle for every (position, dimension) pair.

    positions -- how many positions to encode (sentence length)
    d         -- the encoding size (must match the word-vector size)

    The angle for position 'pos' and dimension 'i' is:
        pos / 10000^(2*(i//2) / d)
    The 10000^... term makes each pair of dimensions oscillate at a
    different frequency: small i -> fast wave, large i -> slow wave.
    """
    pos = np.arange(positions)[:, np.newaxis]          # column of positions
    i = np.arange(d)[np.newaxis, :]                     # row of dimensions
    angle_rates = 1.0 / np.power(10000, (2 * (i // 2)) / np.float32(d))
    return pos * angle_rates                            # shape (positions, d)


def positional_encoding(positions, d):
    """Return the (positions, d) sinusoidal positional-encoding matrix.

    Even dimensions get a SINE of the angle; odd dimensions get a
    COSINE. Pairing sine and cosine at the same frequency is what lets
    the model express a fixed shift in position as a simple rotation -
    which makes relative positions easy to learn.
    """
    angle_rads = get_angles(positions, d)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # even columns -> sin
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # odd columns  -> cos
    return angle_rads


# =================================================================
# 2. Visualizations
# =================================================================
def plot_encoding(pe, filename="positional_encoding.png"):
    """The classic 'barcode' heatmap: position down, dimension across."""
    plt.figure(figsize=(9, 5))
    plt.pcolormesh(pe, cmap="RdBu")
    plt.xlabel("encoding dimension  (d)")
    plt.ylabel("position in sentence")
    plt.title("Sinusoidal positional encoding\n"
              "(each row is one position's unique 'time stamp')")
    plt.colorbar(label="value")
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close()
    print(f"   saved {filename}")


def plot_position_similarity(pe, filename="positional_similarity.png"):
    """Two sanity checks that nearby positions have similar encodings.

    LEFT  -- dot-product similarity between every pair of positions.
             A bright diagonal means each position is most similar to
             itself and to its neighbors.
    RIGHT -- Euclidean distance between every pair of positions. It
             grows smoothly as positions get farther apart - exactly
             the "sense of distance" the transformer needs.
    """
    norm = pe / (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-9)
    corr = norm @ norm.T

    n = pe.shape[0]
    dist = np.zeros((n, n))
    for a in range(n):
        diff = pe - pe[a]
        dist[a] = np.sqrt((diff ** 2).sum(axis=1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    m1 = ax1.pcolormesh(corr, cmap="RdBu")
    ax1.set_title("Similarity between positions\n(bright diagonal = neighbors are alike)")
    ax1.set_xlabel("position"); ax1.set_ylabel("position")
    fig.colorbar(m1, ax=ax1)

    m2 = ax2.pcolormesh(dist, cmap="RdBu")
    ax2.set_title("Distance between positions\n(grows smoothly with separation)")
    ax2.set_xlabel("position"); ax2.set_ylabel("position")
    fig.colorbar(m2, ax=ax2)

    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close()
    print(f"   saved {filename}")


# =================================================================
# MAIN
# =================================================================
def main():
    MAX_POSITIONS = 100     # longest sentence we prepare encodings for
    D_MODEL = 100           # encoding size (matches the word-vector size)

    print("Building the sinusoidal positional encoding ...")
    pe = positional_encoding(MAX_POSITIONS, D_MODEL)
    print(f"   matrix shape: {pe.shape}  (positions x dimensions)")
    print(f"   value range: [{pe.min():.2f}, {pe.max():.2f}]  "
          "(all sines and cosines, so between -1 and 1)")

    print("\nIn a transformer you simply ADD this to the word embeddings:")
    print("   x = word_embedding + positional_encoding[:sequence_length]")
    print("No extra parameters, no training - the pattern is fixed by formula.")

    print("\nDrawing the encoding and the position-similarity checks ...")
    plot_encoding(pe)
    plot_position_similarity(pe)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print("\nQuick check - cosine similarity between position encodings:")
    print(f"   pos 10 vs pos 11 (neighbors) : {cos(pe[10], pe[11]):.3f}")
    print(f"   pos 10 vs pos 50 (far apart) : {cos(pe[10], pe[50]):.3f}")
    print("Neighbors score higher, which is exactly what we want: the model")
    print("can tell 'close' from 'far' just by comparing these encodings.")

    print("\nDone. Open positional_encoding.png (the barcode) and")
    print("positional_similarity.png (the two checks) to see order made visible.")


if __name__ == "__main__":
    main()
