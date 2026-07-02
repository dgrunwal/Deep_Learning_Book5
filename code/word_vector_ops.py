"""
word_vector_ops.py
===================================================================
Load pretrained word vectors and use them to (1) measure how similar
two words are and (2) solve analogies like "man is to woman as king
is to ____".  Pure NumPy -- no deep-learning framework needed.

The big idea
-------------------------------------------------------------------
A word embedding is just a list of numbers (a vector) attached to a
word. Words with similar meanings get similar vectors, so we can do
arithmetic and geometry on MEANING:

  * Similarity  -> measure the ANGLE between two word vectors.
                   Small angle = pointing the same way = similar.
  * Analogy     -> "king - man + woman" lands near "queen", because
                   subtracting "man" and adding "woman" moves you
                   along the same direction that separates every
                   male/female pair.

Where the vectors come from
-------------------------------------------------------------------
Real projects download pretrained vectors such as GloVe
(e.g. the file glove.6B.50d.txt, ~50 numbers per word, trained on
billions of words of text). This script will USE that file if you
put it next to the script. If it is not found, the script falls
back to a tiny built-in set of hand-made vectors so the demo still
runs out of the box and you can see the operations work.

© 2026 David Grunwald. All rights reserved.
"""

import os
import numpy as np


# -------------------------------------------------------------------
# 1. Loading vectors
# -------------------------------------------------------------------
def read_glove_vecs(glove_file):
    """Read a GloVe text file into a {word: vector} dictionary.

    Each line of a GloVe file looks like:
        the 0.418 0.249 -0.412 ...   (word, then its numbers)
    We split each line, take the first token as the word and the
    rest as its vector of floats.
    """
    word_to_vec = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            word_to_vec[word] = np.array(parts[1:], dtype=np.float64)
    return word_to_vec


def tiny_builtin_vectors():
    """A small, hand-built fallback so the script always runs.

    These 6-number vectors are NOT real GloVe vectors -- they are
    toy features (roughly: royalty, gender, age, "is-fruit",
    country-ness, capital-ness) chosen so the demos below work.
    Real embeddings learn features like these automatically; here we
    set them by hand only so you can see the math with no download.
    """
    #          human royal gender  age  fruit country capital
    v = {
        "man":     [1.0, 0.0, -0.6, 0.3,  0.0,  0.0,   0.0],
        "woman":   [1.0, 0.0,  0.6, 0.3,  0.0,  0.0,   0.0],
        "king":    [1.0, 1.0, -0.6, 0.7,  0.0,  0.0,   0.0],
        "queen":   [1.0, 1.0,  0.6, 0.7,  0.0,  0.0,   0.0],
        "boy":     [1.0, 0.0, -0.6, -0.7, 0.0,  0.0,   0.0],
        "girl":    [1.0, 0.0,  0.6, -0.7, 0.0,  0.0,   0.0],
        "father":  [1.0, 0.0, -0.6, 0.8,  0.0,  0.0,   0.0],
        "mother":  [1.0, 0.0,  0.6, 0.8,  0.0,  0.0,   0.0],
        "apple":   [0.0, 0.0,  0.0, 0.0,  1.0,  0.0,   0.0],
        "orange":  [0.0, 0.0,  0.0, 0.0,  1.0,  0.0,   0.0],
        "ball":    [0.0, 0.0,  0.0, 0.0,  0.0,  0.0,   0.0],
        "crocodile":[0.0,0.0,  0.0, 0.2,  0.0,  0.0,   0.0],
        "france":  [0.0, 0.0,  0.0, 0.0,  0.0,  1.0,   0.0],
        "italy":   [0.0, 0.0,  0.0, 0.0,  0.0,  1.0,   0.0],
        "paris":   [0.0, 0.0,  0.0, 0.0,  0.0,  1.0,   1.0],
        "rome":    [0.0, 0.0,  0.0, 0.0,  0.0,  1.0,   1.0],
    }
    return {w: np.array(x, dtype=np.float64) for w, x in v.items()}


def load_vectors():
    """Use a real GloVe file if present, else the built-in toy set."""
    candidates = ["glove.6B.50d.txt",
                  os.path.join("data", "glove.6B.50d.txt")]
    for path in candidates:
        if os.path.exists(path):
            print(f"Loading pretrained vectors from {path} ...")
            return read_glove_vecs(path), True
    print("No GloVe file found -- using a tiny built-in vector set.\n"
          "(Download glove.6B.50d.txt for real results.)\n")
    return tiny_builtin_vectors(), False


# -------------------------------------------------------------------
# 2. Cosine similarity  (the angle between two vectors)
# -------------------------------------------------------------------
def cosine_similarity(u, v):
    """How similar are vectors u and v? Returns a number in [-1, 1].

    Cosine similarity = (u . v) / (|u| * |v|)

      * dot product u.v  -- large when the vectors point the same way
      * |u|, |v|         -- the lengths, which we divide out so that
                            only the DIRECTION (the angle) matters

    Result:  1 = identical direction (very similar),
             0 = perpendicular (unrelated),
            -1 = opposite direction.
    We compare angles, not raw distance, because a word's MEANING is
    in which way its vector points, not how long it is.
    """
    if np.all(u == v):
        return 1.0
    dot = np.dot(u, v)
    norm_u = np.sqrt(np.sum(u ** 2))
    norm_v = np.sqrt(np.sum(v ** 2))
    if np.isclose(norm_u * norm_v, 0.0):
        return 0.0
    return dot / (norm_u * norm_v)


# -------------------------------------------------------------------
# 3. Analogy:  word_a is to word_b as word_c is to ____
# -------------------------------------------------------------------
def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """Solve 'a is to b as c is to ?' with vector arithmetic.

    The trick: the vector (e_b - e_a) is the "direction of change"
    from a to b (for man->woman, that is the gender direction). We
    look for the word whose own change from c, (e_w - e_c), points
    the same way -- i.e. maximizes cosine_similarity(e_b - e_a,
    e_w - e_c).  That is the same as finding e_w closest to
    e_c + (e_b - e_a).
    """
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    e_a, e_b, e_c = (word_to_vec_map[word_a],
                     word_to_vec_map[word_b],
                     word_to_vec_map[word_c])

    target_direction = e_b - e_a          # e.g. woman - man
    best_word, best_sim = None, -np.inf
    for w in word_to_vec_map:
        if w in (word_a, word_b, word_c):  # don't return an input word
            continue
        sim = cosine_similarity(target_direction,
                                word_to_vec_map[w] - e_c)
        if sim > best_sim:
            best_sim, best_word = sim, w
    return best_word


def most_similar(word, word_to_vec_map, k=5):
    """List the k words whose vectors point most like `word`'s."""
    if word not in word_to_vec_map:
        return []
    target = word_to_vec_map[word]
    scored = [(cosine_similarity(target, vec), w)
              for w, vec in word_to_vec_map.items() if w != word]
    scored.sort(reverse=True)
    return scored[:k]


# -------------------------------------------------------------------
# 4. Demo
# -------------------------------------------------------------------
def main():
    word_to_vec_map, is_real = load_vectors()

    print("=== Similarity: how close are two words? ===")
    for a, b in [("father", "mother"), ("ball", "crocodile"),
                 ("man", "woman"), ("apple", "orange")]:
        if a in word_to_vec_map and b in word_to_vec_map:
            s = cosine_similarity(word_to_vec_map[a], word_to_vec_map[b])
            print(f"  cosine_similarity({a:>9}, {b:<9}) = {s:+.3f}")

    print("\n=== Analogy: a is to b as c is to ___ ===")
    analogies = [("man", "woman", "king"),
                 ("man", "boy", "woman"),
                 ("italy", "rome", "france")]
    for a, b, c in analogies:
        if all(w in word_to_vec_map for w in (a, b, c)):
            answer = complete_analogy(a, b, c, word_to_vec_map)
            print(f"  {a} -> {b}  ::  {c} -> {answer}")

    print("\n=== Nearest neighbors (most similar words) ===")
    for word in ["king", "apple"]:
        nbrs = most_similar(word, word_to_vec_map, k=3)
        shown = ", ".join(f"{w} ({s:+.2f})" for s, w in nbrs)
        print(f"  {word:>6}: {shown}")

    if not is_real:
        print("\n(These used the tiny built-in vectors. With real GloVe "
              "vectors\n the same code gives richer, more accurate results.)")


if __name__ == "__main__":
    main()
