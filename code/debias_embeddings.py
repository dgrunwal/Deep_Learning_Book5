"""
debias_embeddings.py
==================================================================
Companion script for Chapter 6 of "Deep Learning for Beginners:
Sequence Models" (AI Foundations Series, Book 5).

WHAT THIS SCRIPT DOES
-----------------------------------------------------------------
Word vectors learn meaning from human text - and human text carries
human bias. This script MEASURES gender bias in pretrained word
vectors and then REDUCES it, using the algorithm of Bolukbasi et
al. (2016). It mirrors the Course-5 "debiasing" lab.

It walks through four ideas, in order:

  1. COSINE SIMILARITY - a number from -1 to 1 saying how aligned
     two vectors are. This is our measuring stick.

  2. THE BIAS DIRECTION g - we build a single "gender axis" by
     subtracting:   g = vector("woman") - vector("man").
     Projecting any word onto g tells us how "gendered" it is.

  3. NEUTRALIZE - for words that SHOULD be gender-neutral (like
     "receptionist" or "engineer"), we zero out their component
     along g, so they sit squarely on the gender-neutral plane.

  4. EQUALIZE - for words that SHOULD stay a gendered pair (like
     "actor"/"actress"), we make them perfectly symmetric about
     that neutral plane, so neither leans toward other neutral words.

To keep the script self-contained, it builds a tiny illustrative set
of vectors. The CODE is exactly what you would run on real GloVe; only
the vocabulary is shrunk. Swap in read_glove_vecs('glove.6B.50d.txt')
to run it on the real 400,000-word table.
==================================================================
"""

import numpy as np


# -----------------------------------------------------------------
# 1. COSINE SIMILARITY  - our measuring stick
# -----------------------------------------------------------------
def cosine_similarity(u, v):
    """How aligned are two vectors? Returns a number in [-1, 1].

         +1  -> point the same way (very similar)
          0  -> perpendicular (unrelated)
         -1  -> point opposite ways

    Formula:  (u . v) / (||u|| * ||v||)
    It is the dot product divided by both lengths, which cancels out
    how LONG the vectors are and leaves only their DIRECTION.
    """
    if np.all(u == v):
        return 1.0
    dot = np.dot(u, v)
    norm_u = np.sqrt(np.sum(u ** 2))
    norm_v = np.sqrt(np.sum(v ** 2))
    if np.isclose(norm_u * norm_v, 0, atol=1e-32):
        return 0.0
    return dot / (norm_u * norm_v)


# -----------------------------------------------------------------
# 3. NEUTRALIZE  - remove the gender component from a neutral word
# -----------------------------------------------------------------
def neutralize(word, g, word_to_vec_map):
    """Project a word vector onto the space orthogonal to g.

    The "bias component" of a word e along the gender axis g is:

        e_biascomponent = (e . g) / (g . g) * g

    That is the shadow e casts on the g axis. Subtract it and what
    remains has ZERO gender component - it has been neutralized.

        e_debiased = e - e_biascomponent
    """
    e = word_to_vec_map[word]
    e_biascomponent = (np.dot(e, g) / np.sum(g ** 2)) * g
    e_debiased = e - e_biascomponent
    return e_debiased


# -----------------------------------------------------------------
# 4. EQUALIZE  - make a gendered pair symmetric about the neutral plane
# -----------------------------------------------------------------
def equalize(pair, bias_axis, word_to_vec_map):
    """Make two words (e.g. 'actor','actress') mirror images across g.

    After this, the two words differ ONLY in their gender component and
    are exactly the same distance from every neutralized word. The math
    (Bolukbasi et al., 2016):

        mu        = (e_w1 + e_w2) / 2           # midpoint of the pair
        mu_B      = projection of mu onto bias_axis
        mu_orth   = mu - mu_B                    # the neutral part they share

        e_w1B     = projection of e_w1 onto bias_axis
        e_w2B     = projection of e_w2 onto bias_axis

        # rescale the gender parts so the pair is symmetric:
        corrected_e_w1B = sqrt(|1 - ||mu_orth||^2|) * (e_w1B - mu_B)/||e_w1B - mu_B||
        corrected_e_w2B = sqrt(|1 - ||mu_orth||^2|) * (e_w2B - mu_B)/||e_w2B - mu_B||

        e1 = corrected_e_w1B + mu_orth
        e2 = corrected_e_w2B + mu_orth
    """
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    mu = (e_w1 + e_w2) / 2.0
    mu_B = (np.dot(mu, bias_axis) / np.sum(bias_axis ** 2)) * bias_axis
    mu_orth = mu - mu_B

    e_w1B = (np.dot(e_w1, bias_axis) / np.sum(bias_axis ** 2)) * bias_axis
    e_w2B = (np.dot(e_w2, bias_axis) / np.sum(bias_axis ** 2)) * bias_axis

    scale = np.sqrt(np.abs(1.0 - np.sum(mu_orth ** 2)))
    corrected_e_w1B = scale * (e_w1B - mu_B) / (np.linalg.norm(e_w1B - mu_B) + 1e-12)
    corrected_e_w2B = scale * (e_w2B - mu_B) / (np.linalg.norm(e_w2B - mu_B) + 1e-12)

    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth
    return e1, e2


# -----------------------------------------------------------------
# A tiny illustrative vector set (stands in for real GloVe)
# -----------------------------------------------------------------
def build_demo_vectors(dim=50, seed=3):
    """Build vectors where a clear, deliberate gender axis exists.

    Real GloVe gets its (unfortunate) gender lean from training text.
    Here we plant it on purpose so the before/after numbers are vivid.
    """
    rng = np.random.default_rng(seed)
    gender = np.zeros(dim)
    gender[0] = 1.0                                  # axis 0 = "gender" by construction
    base = {w: rng.normal(0, 0.4, dim) for w in [
        "receptionist", "engineer", "nurse", "scientist", "babysit",
        "actor", "technology", "literature"]}

    wv = {}
    wv["man"] = -1.0 * gender + rng.normal(0, 0.05, dim)
    wv["woman"] = 1.0 * gender + rng.normal(0, 0.05, dim)
    # Neutral jobs given a SMALL unwanted gender lean (the bias we remove):
    wv["receptionist"] = base["receptionist"] + 0.35 * gender
    wv["engineer"] = base["engineer"] - 0.35 * gender
    wv["nurse"] = base["nurse"] + 0.30 * gender
    wv["scientist"] = base["scientist"] - 0.30 * gender
    wv["technology"] = base["technology"] - 0.28 * gender
    wv["literature"] = base["literature"] + 0.28 * gender
    wv["babysit"] = base["babysit"] + 0.20 * gender
    # A pair we want to KEEP gendered but make symmetric:
    wv["actor"] = base["actor"] - 0.9 * gender
    wv["actress"] = base["actor"] + 0.9 * gender
    return wv


# =================================================================
# MAIN
# =================================================================
def main():
    word_to_vec_map = build_demo_vectors()

    # ---- 2. Build the gender bias direction ----
    g = word_to_vec_map["woman"] - word_to_vec_map["man"]
    print("=" * 60)
    print("STEP 2: the gender axis  g = vector('woman') - vector('man')")
    print("=" * 60)
    print("Cosine similarity of each NEUTRAL job with the gender axis g.")
    print("(A job that should be neutral but isn't will be far from 0.)\n")
    for w in ["receptionist", "engineer", "nurse", "scientist",
              "technology", "literature"]:
        print(f"   {w:14s}: {cosine_similarity(word_to_vec_map[w], g):+.3f}")

    # ---- 3. Neutralize one neutral word ----
    print("\n" + "=" * 60)
    print("STEP 3: NEUTRALIZE a word that should be gender-neutral")
    print("=" * 60)
    before = cosine_similarity(word_to_vec_map["receptionist"], g)
    e_after = neutralize("receptionist", g, word_to_vec_map)
    after = cosine_similarity(e_after, g)
    print(f"   receptionist . g  BEFORE neutralizing : {before:+.5f}")
    print(f"   receptionist . g  AFTER  neutralizing : {after:+.1e}")
    print("   -> after neutralizing it is essentially 0: no gender lean left.")

    # ---- 4. Equalize a gendered pair ----
    print("\n" + "=" * 60)
    print("STEP 4: EQUALIZE the pair ('actor', 'actress')")
    print("=" * 60)
    print("Before: the two may sit at different distances from neutral words.")
    print(f"   cosine(actor,   g) = {cosine_similarity(word_to_vec_map['actor'], g):+.3f}")
    print(f"   cosine(actress, g) = {cosine_similarity(word_to_vec_map['actress'], g):+.3f}")
    e1, e2 = equalize(("actor", "actress"), g, word_to_vec_map)
    print("After equalizing (should be equal and opposite):")
    print(f"   cosine(actor,   g) = {cosine_similarity(e1, g):+.3f}")
    print(f"   cosine(actress, g) = {cosine_similarity(e2, g):+.3f}")

    print("\nDone. We measured bias with cosine similarity, removed it from")
    print("neutral words with neutralize(), and made a true pair symmetric")
    print("with equalize(). Real debiasing is harder - bias hides in many")
    print("directions - but the same tools are the honest starting point.")


if __name__ == "__main__":
    main()
