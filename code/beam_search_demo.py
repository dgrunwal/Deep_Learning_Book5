"""
beam_search_demo.py
==================================================================
Companion script for Chapter 7 of "Deep Learning for Beginners:
Sequence Models" (AI Foundations Series, Book 5).

WHAT THIS SCRIPT DOES
-----------------------------------------------------------------
It shows how BEAM SEARCH decodes a sentence from a tiny language
model, and how changing the BEAM WIDTH changes the answer.

The big idea in one line:
    A decoder produces one word at a time, each with a probability.
    We want the whole SENTENCE with the highest probability - but
    checking every possible sentence is impossible, so beam search
    keeps only the best few partial sentences as it goes.

To keep everything visible and dependency-free, we do NOT train a
real neural network. Instead we hand-write a tiny "decoder" as a
lookup table of next-word probabilities. That is exactly the kind
of number a real decoder's softmax would produce at each step -
we are just supplying them directly so the search is easy to watch.

You will see three things:
    1. GREEDY decoding (always take the single best next word) -
       which can paint itself into a corner.
    2. BEAM SEARCH with width B, keeping the best B sentences.
    3. How a wider beam can find a better overall sentence, and how
       LENGTH NORMALIZATION stops the search preferring short ones.
==================================================================
"""

import math

# -----------------------------------------------------------------
# 1. A tiny toy "decoder"
# -----------------------------------------------------------------
# A real decoder is a neural network: given the words chosen so far,
# it outputs a probability for every possible next word. We fake that
# with a dictionary. The KEY is the sequence chosen so far (a tuple of
# words); the VALUE is a dict of {next_word: probability}.
#
# "<s>"  marks the start of the sentence.
# "</s>" marks the end   of the sentence (the decoder wants to stop).
#
# The numbers below are rigged to make one classic point: the word that
# looks best FIRST ("the") leads only to mediocre endings, while a word
# that looks slightly worse first ("a") leads to the best whole sentence.
# Greedy decoding grabs "the" and misses the better sentence; a wide
# enough beam keeps "a" alive long enough to discover it.

DECODER = {
    # First word: "the" looks best, "a" is close behind.
    ("<s>",): {"the": 0.5, "a": 0.45, "cats": 0.05},

    # --- The "the" branch leads to a SHORT, high-probability sentence. ---
    # "the dog" is only two words but scores well, so greedy takes it and
    # raw beam search over-prefers it - the classic "too short" trap.
    ("<s>", "the"): {"dog": 1.0},
    ("<s>", "the", "dog"): {"</s>": 1.0},               # p = 0.5 * 1.0 = 0.50 (SHORT)

    # --- The "a" branch hides the best whole sentence, but it is longer. ---
    ("<s>", "a"): {"happy": 0.85, "sad": 0.15},
    ("<s>", "a", "happy"): {"cat": 0.9, "dog": 0.1},
    ("<s>", "a", "happy", "cat"): {"purred": 1.0},
    ("<s>", "a", "happy", "cat", "purred"): {"</s>": 1.0},  # p = 0.45*0.85*0.9 = 0.344
    ("<s>", "a", "sad"): {"cat": 1.0},
    ("<s>", "a", "sad", "cat"): {"</s>": 1.0},

    ("<s>", "cats",): {"sleep": 1.0},
    ("<s>", "cats", "sleep"): {"</s>": 1.0},
}
# Probability summary (why each lesson works):
#   "the dog"            = 0.500   <- greedy lands here; SHORT and high-prob
#   "a happy cat purred" = 0.344   <- longer, lower raw prob, but the better
#                                      sentence a human would prefer
# Raw beam search still ranks "the dog" first (0.500 > 0.344). LENGTH
# NORMALIZATION divides by length, which lifts the longer sentence above
# the short one - flipping the winner to "a happy cat purred".


def next_word_probs(sequence):
    """Return the decoder's {next_word: probability} for a sequence.

    A real network would compute this with a forward pass; we just look
    it up. If a sequence isn't in the table, the sentence must end.
    """
    return DECODER.get(tuple(sequence), {"</s>": 1.0})


# -----------------------------------------------------------------
# 2. Greedy decoding - always take the single best next word
# -----------------------------------------------------------------
def greedy_decode(max_len=8):
    """Build a sentence by always grabbing the most likely next word.

    Fast, but short-sighted: it never reconsiders. One tempting first
    word can lead somewhere mediocre, and greedy can't back out.
    """
    seq = ["<s>"]
    logprob = 0.0
    while len(seq) < max_len:
        probs = next_word_probs(seq)
        word = max(probs, key=probs.get)      # the single best next word
        logprob += math.log(probs[word])
        seq.append(word)
        if word == "</s>":
            break
    return seq, logprob


# -----------------------------------------------------------------
# 3. Beam search - keep the best B partial sentences at every step
# -----------------------------------------------------------------
# Why add LOG-probabilities instead of multiplying probabilities?
# A sentence's probability is p(w1) * p(w2) * ... Multiplying many
# numbers below 1 underflows toward zero and is hard to compare.
# Taking logs turns the product into a SUM (log(ab) = log a + log b),
# which is numerically safe and keeps the same ranking.

def beam_search(beam_width, max_len=8, length_norm=False, alpha=0.7):
    """Decode with beam search.

    beam_width  -- how many partial sentences (B) to keep alive.
    length_norm -- if True, divide each score by length^alpha so the
                   search doesn't unfairly prefer very short sentences.

    Returns the finished sentences it found, best first.
    """
    # Each beam entry is (sequence, cumulative_logprob, finished?).
    beams = [(["<s>"], 0.0, False)]
    finished = []

    for _ in range(max_len):
        candidates = []
        for seq, score, done in beams:
            if done:                                   # already ended - carry it forward
                candidates.append((seq, score, True))
                continue
            for word, p in next_word_probs(seq).items():
                new_seq = seq + [word]
                new_score = score + math.log(p)        # add log-prob
                candidates.append((new_seq, new_score, word == "</s>"))

        # Rank all candidates. With length normalization, longer sentences
        # are not penalized just for having more (sub-1) factors multiplied in.
        def rank_key(item):
            seq, score, _ = item
            if length_norm:
                length = max(len([w for w in seq if w not in ("<s>", "</s>")]), 1)
                return score / (length ** alpha)
            return score

        candidates.sort(key=rank_key, reverse=True)
        beams = candidates[:beam_width]                # KEEP ONLY THE TOP B

        # Move any finished sentences aside; stop early if all beams are done.
        finished += [(s, sc) for (s, sc, d) in beams if d]
        if all(d for (_, _, d) in beams):
            break

    # Collect every finished sentence we saw, ranked the same way.
    all_done = finished + [(s, sc) for (s, sc, d) in beams if d]
    seen, unique = set(), []
    for s, sc in all_done:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique.append((s, sc))

    def final_key(item):
        seq, score = item
        if length_norm:
            length = max(len([w for w in seq if w not in ("<s>", "</s>")]), 1)
            return score / (length ** alpha)
        return score

    unique.sort(key=final_key, reverse=True)
    return unique


# -----------------------------------------------------------------
# helpers for pretty printing
# -----------------------------------------------------------------
def clean(seq):
    """Drop the <s> / </s> markers for readable output."""
    return " ".join(w for w in seq if w not in ("<s>", "</s>"))


def prob_of(seq):
    """Recover the plain sentence probability from the decoder."""
    p, partial = 1.0, ["<s>"]
    for w in seq[1:]:
        p *= next_word_probs(partial).get(w, 0.0)
        partial.append(w)
    return p


# =================================================================
# MAIN
# =================================================================
def main():
    print("=" * 60)
    print("GREEDY DECODING (take the single best word each step)")
    print("=" * 60)
    g_seq, _ = greedy_decode()
    print(f'   result : "{clean(g_seq)}"')
    print(f'   prob   : {prob_of(g_seq):.4f}')
    print('   Greedy took the best first word ("the") and ended up with')
    print('   the short "the dog". It never explored the "a ..." path.')

    print("\n" + "=" * 60)
    print("BEAM SEARCH at different beam widths B")
    print("=" * 60)
    for B in [1, 2, 3]:
        best = beam_search(beam_width=B)
        seq, score = best[0]
        note = "  (B=1 is just greedy)" if B == 1 else ""
        print(f"\n   Beam width B = {B}{note}")
        print(f'      top sentence  : "{clean(seq)}"  (prob {prob_of(seq):.3f})')
        if len(best) > 1:
            others = ", ".join(f'"{clean(s)}"' for s, _ in best[1:3])
            print(f'      also alive    : {others}')

    print("\n   A wider beam keeps more sentences alive. Here it discovers")
    print('   "a happy cat purred" and holds onto it - even though its raw')
    print('   probability (0.344) is below the short "the dog" (0.500).')
    print("   Raw scoring still ranks the short one first. That is the")
    print("   problem length normalization fixes next.")

    print("\n" + "=" * 60)
    print("LENGTH NORMALIZATION")
    print("=" * 60)
    print("Every extra word multiplies in another below-1 probability, so")
    print("raw scores secretly favor SHORT sentences. Dividing each score")
    print("by length^alpha levels the field. Same beam (B=3), both ways:")
    raw = beam_search(beam_width=3, length_norm=False)[0]
    norm = beam_search(beam_width=3, length_norm=True)[0]
    print(f'\n   without normalization : "{clean(raw[0])}"   (short one wins)')
    print(f'   with    normalization : "{clean(norm[0])}"   (better one wins)')

    print("\nDone. Three lessons: greedy is short-sighted; a wider beam")
    print("searches more of the space; and length normalization stops the")
    print("search from settling for a sentence that is merely short.")


if __name__ == "__main__":
    main()
