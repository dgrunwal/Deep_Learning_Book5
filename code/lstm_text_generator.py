r"""
lstm_text_generator.py
===================================================================
A character-level LSTM that LEARNS to generate text, written from
scratch in pure NumPy. No TensorFlow, no PyTorch -- every gate is
visible so you can watch the LSTM's memory work.

What "character-level" means
-------------------------------------------------------------------
Instead of predicting the next WORD, this model predicts the next
LETTER. You feed it a string like "hello worl" and it learns to
guess "d". Do that one character at a time and the model can write
brand-new text that imitates whatever you trained it on.

Why an LSTM and not a plain RNN?
-------------------------------------------------------------------
A plain RNN forgets. Over many time steps its gradient vanishes, so
the start of a sentence stops influencing the end (see Chapter 3).
An LSTM fixes this with a MEMORY CELL (c) that runs along the top of
the network like a conveyor belt, plus three GATES that learn to
control it:

    forget gate (f) -- how much of the old memory to KEEP
    input  gate (i) -- how much of the new candidate to WRITE
    output gate (o) -- how much of the memory to REVEAL as output

The one equation to remember is the cell update:

        c<t> = f * c<t-1>  +  i * c-tilde<t>
               \_______/     \___________/
                keep old       write new

When the forget gate f is near 1 and the input gate i is near 0, the
cell just copies itself forward: c<t> = c<t-1>. That near-untouched
highway is what lets memory (and gradient) travel across long
distances without vanishing. THIS is the idea to internalize: if you
only deeply understand one cell, make it the LSTM. The same "let
information flow along a protected path, and gate what gets added"
idea reappears, in spirit, all the way up to the transformer.

This file trains a small LSTM on a built-in sample text and then
generates new text. It is intentionally tiny so it runs on a laptop
CPU in under a minute.

© 2026 David Grunwald. All rights reserved.
"""

import numpy as np


# -------------------------------------------------------------------
# Small activation helpers
# -------------------------------------------------------------------
def sigmoid(z):
    """Squash any number into the range 0..1. A gate uses this so its
    output reads like a dial: 0 = 'closed', 1 = 'fully open'."""
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    """Turn raw scores into probabilities over the alphabet."""
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


# -------------------------------------------------------------------
# The LSTM
# -------------------------------------------------------------------
class CharLSTM:
    """A minimal character-level LSTM trained with plain SGD.

    Sizes:
      vocab_size (V) -- number of distinct characters
      hidden (H)     -- size of the hidden state and memory cell
    """

    def __init__(self, vocab_size, hidden=64, seq_len=25, lr=0.1, seed=1):
        self.V = vocab_size
        self.H = hidden
        self.seq_len = seq_len      # how many characters per training chunk
        self.lr = lr
        rng = np.random.RandomState(seed)

        # One stacked weight matrix produces all four gate pre-values
        # at once: forget, input, candidate, output. Each is (H, ...).
        # We concatenate [a_prev; x] so a single matrix mixes the
        # previous hidden state with the current character.
        Z = hidden + vocab_size
        s = 0.1
        # Wf, Wi, Wc, Wo stacked vertically -> shape (4H, H+V)
        self.W = rng.randn(4 * hidden, Z) * s
        self.b = np.zeros((4 * hidden, 1))
        self.b[:hidden] = 1.0       # start forget-gate bias high so the
        #                              cell KEEPS memory early in training
        # Output projection from hidden state to a score per character.
        self.Wy = rng.randn(vocab_size, hidden) * s
        self.by = np.zeros((vocab_size, 1))

    # ----- one forward step through the gates -----
    def step(self, x, a_prev, c_prev):
        """Run one character through the LSTM cell.

        Returns the new hidden state a, new memory cell c, the
        prediction probabilities p, and a cache for backprop.
        """
        H = self.H
        z = np.vstack((a_prev, x))          # stack memory + input
        gates = self.W @ z + self.b         # all four gate pre-values

        f = sigmoid(gates[0:H])             # forget gate  (keep old c?)
        i = sigmoid(gates[H:2 * H])         # input gate   (write new?)
        c_bar = np.tanh(gates[2 * H:3 * H])  # candidate memory
        o = sigmoid(gates[3 * H:4 * H])     # output gate  (reveal c?)

        c = f * c_prev + i * c_bar          # THE cell update
        a = o * np.tanh(c)                  # hidden state shown to world

        y = self.Wy @ a + self.by
        p = softmax(y)                      # next-character probabilities

        cache = (z, f, i, c_bar, o, c, c_prev, a, p)
        return a, c, p, cache

    # ----- forward + backward over one sequence -----
    def loss_and_grads(self, inputs, targets, a_prev, c_prev):
        """Forward across the chunk, then backprop through time.

        inputs/targets are lists of character indices. Returns the
        loss, the gradients, and the final (a, c) to carry on with.
        """
        H, V = self.H, self.V
        xs, caches = {}, {}
        a, c = a_prev, c_prev
        loss = 0.0

        # ---- forward ----
        for t in range(len(inputs)):
            x = np.zeros((V, 1)); x[inputs[t]] = 1     # one-hot char
            a, c, p, caches[t] = self.step(x, a, c)
            xs[t] = x
            loss += -np.log(p[targets[t], 0] + 1e-9)   # cross-entropy

        # ---- backward (through time) ----
        dW = np.zeros_like(self.W); db = np.zeros_like(self.b)
        dWy = np.zeros_like(self.Wy); dby = np.zeros_like(self.by)
        da_next = np.zeros((H, 1)); dc_next = np.zeros((H, 1))

        for t in reversed(range(len(inputs))):
            (z, f, i, c_bar, o, c_t, c_prev_t, a_t, p) = caches[t]

            dy = p.copy(); dy[targets[t]] -= 1          # softmax+CE grad
            dWy += dy @ a_t.T; dby += dy
            da = self.Wy.T @ dy + da_next               # into hidden state

            do = da * np.tanh(c_t)
            dc = da * o * (1 - np.tanh(c_t) ** 2) + dc_next
            df = dc * c_prev_t
            di = dc * c_bar
            dc_bar = dc * i

            # back through the gate nonlinearities
            df_raw = df * f * (1 - f)
            di_raw = di * i * (1 - i)
            do_raw = do * o * (1 - o)
            dc_bar_raw = dc_bar * (1 - c_bar ** 2)

            dgates = np.vstack((df_raw, di_raw, dc_bar_raw, do_raw))
            dW += dgates @ z.T; db += dgates

            dz = self.W.T @ dgates
            da_next = dz[:H]                             # to earlier step
            dc_next = f * dc                             # cell highway grad

        # clip to tame exploding gradients (Chapter 3's quick fix)
        for g in (dW, db, dWy, dby):
            np.clip(g, -5, 5, out=g)

        return loss, (dW, db, dWy, dby), a, c

    def sgd_update(self, grads):
        dW, db, dWy, dby = grads
        self.W -= self.lr * dW; self.b -= self.lr * db
        self.Wy -= self.lr * dWy; self.by -= self.lr * dby

    # ----- generate fresh text -----
    def sample(self, seed_idx, n, a=None, c=None):
        """Generate n characters, feeding each prediction back in."""
        if a is None: a = np.zeros((self.H, 1))
        if c is None: c = np.zeros((self.H, 1))
        x = np.zeros((self.V, 1)); x[seed_idx] = 1
        out = [seed_idx]
        for _ in range(n):
            a, c, p, _ = self.step(x, a, c)
            idx = np.random.choice(range(self.V), p=p.ravel())
            x = np.zeros((self.V, 1)); x[idx] = 1
            out.append(idx)
        return out


def main():
    # A tiny training text. Swap in any string (or a file) you like.
    text = ("the cat sat on the mat. the cat ate the fish. "
            "a dog ran to the cat. the cat was on the mat again. ") * 6

    chars = sorted(set(text))
    V = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = [stoi[ch] for ch in text]

    model = CharLSTM(vocab_size=V, hidden=64, seq_len=25, lr=0.1)

    print("Training a character-level LSTM on a tiny text...\n")
    a = np.zeros((model.H, 1)); c = np.zeros((model.H, 1))
    p = 0
    smooth = -np.log(1.0 / V) * model.seq_len   # starting loss guess
    for step in range(2001):
        # reset the running state when we reach the end of the text
        if p + model.seq_len + 1 >= len(data) or step == 0:
            a = np.zeros((model.H, 1)); c = np.zeros((model.H, 1)); p = 0
        inputs = data[p:p + model.seq_len]
        targets = data[p + 1:p + model.seq_len + 1]

        loss, grads, a, c = model.loss_and_grads(inputs, targets, a, c)
        model.sgd_update(grads)
        smooth = 0.999 * smooth + 0.001 * loss
        p += model.seq_len

        if step % 500 == 0:
            seed = stoi['t']
            gen = "".join(itos[i] for i in model.sample(seed, 60))
            print(f"step {step:4d} | loss {smooth:6.2f} | sample: {gen!r}")

    print("\nAs the loss falls, the samples drift from noise toward the")
    print("patterns in the training text -- the gates have learned what")
    print("to keep and what to forget.")


if __name__ == "__main__":
    main()
