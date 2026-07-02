"""
rnn_step_forward.py
===================================================================
ONE recurrent neural network (RNN) time step, written in pure NumPy.

No TensorFlow. No PyTorch. No Keras. Nothing is hidden.

Why does this script exist?
---------------------------------------------------------------
A frameworks-based RNN does all of its real work inside a loop over
time that you never see. That loop is the whole idea of an RNN, so
in Book 5 we build it by hand once, with our own eyes on every
number, before we ever let a framework run it for us.

This file computes a SINGLE step of that loop. Read it top to
bottom and you will understand the recurrence relation that every
RNN, GRU, LSTM, and even the attention models later in the book are
built on top of.

The recurrence in one line of math:

        a<t> = g( Waa . a<t-1> + Wax . x<t> + ba )
        y<t> = softmax( Wya . a<t> + by )

In words: to produce the new hidden state a<t>, mix together
(1) what you remembered from the previous step, a<t-1>, and
(2) what you are reading right now, x<t>. Squash the mix with an
activation (we use tanh). That new state is BOTH the memory you
hand to the next step AND the basis for this step's prediction y<t>.

© 2026 David Grunwald. All rights reserved.
"""

import numpy as np


def softmax(z):
    """Turn a vector of raw scores into probabilities that sum to 1.

    We subtract the max first (z - np.max(z)) only for numerical
    safety: it prevents np.exp from overflowing on large inputs and
    does not change the result. The output is a probability
    distribution over the vocabulary -- one number per possible
    next word, all positive, all adding up to 1.0.
    """
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def rnn_step_forward(x_t, a_prev, parameters):
    """Run the RNN forward by exactly ONE time step.

    Inputs
    ------
    x_t : np.ndarray, shape (n_x, 1)
        The input at this time step. For text this is usually a
        one-hot column vector: a column of zeros with a single 1 in
        the row of the current word. n_x is the vocabulary size.

    a_prev : np.ndarray, shape (n_a, 1)
        The hidden state handed forward from the PREVIOUS step
        (a<t-1>). At the very first step this is just zeros -- the
        network starts with a blank memory. n_a is how many numbers
        we use to summarize everything seen so far.

    parameters : dict
        The weights and biases. Critically, these are the SAME
        objects at every time step -- an RNN reuses one small set of
        parameters across the whole sequence. That parameter sharing
        is what lets it handle sequences of any length.
          Wax : (n_a, n_x)  weights on the current input x<t>
          Waa : (n_a, n_a)  weights on the previous state a<t-1>
          Wya : (n_y, n_a)  weights from state to output
          ba  : (n_a, 1)    bias for the hidden state
          by  : (n_y, 1)    bias for the output

    Returns
    -------
    a_next : np.ndarray, shape (n_a, 1)
        The NEW hidden state a<t>. This is the memory we will pass
        into the next call as a_prev. The recurrence lives right
        here: today's output becomes tomorrow's input.

    y_pred : np.ndarray, shape (n_y, 1)
        The prediction for this step as a probability distribution
        (for language models, the predicted next word).
    """
    # Unpack the shared parameters.
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # --- The recurrence, line by line ---------------------------
    # 1. Listen to the current input:        Wax . x<t>
    # 2. Recall the previous state:          Waa . a<t-1>
    # 3. Add them, add the bias, squash with tanh so the state
    #    stays in a stable range of -1..1.   a<t> = tanh(...)
    a_next = np.tanh(np.dot(Wax, x_t) + np.dot(Waa, a_prev) + ba)

    # 4. Read a prediction off the new state and normalize it
    #    into probabilities.                 y<t> = softmax(Wya . a<t> + by)
    y_pred = softmax(np.dot(Wya, a_next) + by)

    return a_next, y_pred


def main():
    """A tiny, fully concrete run so the shapes are visible.

    We invent small dimensions and random weights, build one input
    column, and step the RNN once. The point is not the numbers
    themselves but seeing that every array has the shape the math
    promised.
    """
    # Reproducible random numbers so the output is the same each run.
    np.random.seed(1)

    # Toy sizes. Pick them small enough to reason about by hand.
    n_x = 5   # vocabulary size  -> length of each input vector
    n_a = 3   # hidden-state size -> how much the RNN can remember
    n_y = 5   # output size       -> next-word distribution over the vocab

    # The input x<t>: a one-hot vector standing for "word index 2".
    x_t = np.zeros((n_x, 1))
    x_t[2, 0] = 1.0

    # The previous hidden state. At the start of a sequence this is
    # all zeros -- the network has not seen anything yet.
    a_prev = np.zeros((n_a, 1))

    # Random starting weights. In a real model these are LEARNED by
    # training; here we only want to watch the forward step work.
    parameters = {
        "Wax": np.random.randn(n_a, n_x),
        "Waa": np.random.randn(n_a, n_a),
        "Wya": np.random.randn(n_y, n_a),
        "ba": np.random.randn(n_a, 1),
        "by": np.random.randn(n_y, 1),
    }

    a_next, y_pred = rnn_step_forward(x_t, a_prev, parameters)

    print("=== One RNN time step ===\n")
    print("Input x<t> shape      :", x_t.shape, "(n_x, 1)")
    print("Prev state a<t-1> shape:", a_prev.shape, "(n_a, 1)\n")

    print("New hidden state a<t>  (n_a, 1):")
    print(a_next, "\n")

    print("Prediction y<t>        (n_y, 1):")
    print(y_pred)
    print("\nProbabilities sum to   :", round(float(np.sum(y_pred)), 6),
          "(should be 1.0)")

    # Show the recurrence by hand: feed a<t> back in as the new
    # a_prev for a second, imaginary step. This is the loop an RNN
    # runs across an entire sequence.
    x_next = np.zeros((n_x, 1))
    x_next[4, 0] = 1.0
    a_next2, _ = rnn_step_forward(x_next, a_next, parameters)
    print("\nFeeding a<t> back in as memory for step t+1 gives a new")
    print("state of shape", a_next2.shape, "- that feedback loop IS the RNN.")


if __name__ == "__main__":
    main()
