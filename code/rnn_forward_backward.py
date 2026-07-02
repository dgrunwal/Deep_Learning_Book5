"""
rnn_forward_backward.py
===================================================================
A COMPLETE forward AND backward pass of a recurrent neural network
over a whole sequence, written in pure NumPy. No TensorFlow, no
PyTorch -- every gradient is computed by hand so you can see exactly
what "backpropagation through time" means.

How this builds on Chapter 1
-------------------------------------------------------------------
In Chapter 1 (rnn_step_forward.py) we ran a SINGLE time step. Here
we do two new things:

  1. FORWARD over the whole sequence: loop the same RNN cell across
     all time steps, carrying the hidden state forward and saving a
     "cache" at each step so we can use it on the way back.

  2. BACKWARD over the whole sequence: start at the LAST time step
     and walk to the FIRST, computing how much each weight should
     change. Because every time step shares the SAME weights, the
     gradients from every step are ADDED together. That summing is
     the whole idea of backprop through time.

The shapes (read these once and the code is easy)
-------------------------------------------------------------------
  n_x = vocabulary size (length of each one-hot input vector)
  n_a = hidden-state size (how much the RNN can remember)
  n_y = output size (length of each prediction vector)
  m   = number of examples processed at once (the "batch")
  T_x = number of time steps in the sequence

  x   : (n_x, m, T_x)   the whole input sequence
  a   : (n_a, m, T_x)   the hidden state at every step
  y   : (n_y, m, T_x)   the prediction at every step

© 2026 David Grunwald. All rights reserved.
"""

import numpy as np


def softmax(z):
    """Turn raw scores into probabilities that sum to 1 (per column)."""
    e = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)


# ===================================================================
# FORWARD PASS
# ===================================================================

def rnn_cell_forward(xt, a_prev, parameters):
    """ONE forward time step (same as Chapter 1, now batch-aware).

    Returns the new hidden state, the prediction, and a `cache` --
    a small bundle of the values the backward pass will need later.
    Saving work on the way forward so we can reuse it on the way
    back is the trick that makes backprop efficient.
    """
    Wax, Waa, Wya = parameters["Wax"], parameters["Waa"], parameters["Wya"]
    ba, by = parameters["ba"], parameters["by"]

    # The recurrence: mix previous memory with the current input,
    # then squash with tanh to keep the state in -1..1.
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    # Read a prediction off the new state.
    yt_pred = softmax(np.dot(Wya, a_next) + by)

    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    """Run the RNN forward across EVERY time step of the sequence.

    We loop t = 0, 1, 2, ... T_x-1. At each step the hidden state we
    just produced (a_next) becomes the previous state for the next
    step. We stash every step's prediction and cache so the backward
    pass has everything it needs.
    """
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # Pre-allocate room for every step's hidden state and prediction.
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    a_next = a0  # start with the initial (usually zero) memory
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        a[:, :, t] = a_next       # remember the state at step t
        y_pred[:, :, t] = yt_pred  # remember the prediction at step t
        caches.append(cache)

    caches = (caches, x)
    return a, y_pred, caches


# ===================================================================
# BACKWARD PASS  (backpropagation through time)
# ===================================================================

def rnn_cell_backward(da_next, cache):
    """ONE backward time step.

    Given da_next -- how much the loss wants this step's hidden state
    to change -- we work out how much each weight, the input, and the
    PREVIOUS hidden state should change. That last piece (da_prev) is
    what we hand to the step before us, so the signal keeps flowing
    backward through time.

    The only calculus you need: the derivative of tanh(z) is
    (1 - tanh(z)^2). Since a_next = tanh(z), that is just
    (1 - a_next**2). Everything else is the chain rule.
    """
    (a_next, a_prev, xt, parameters) = cache
    Wax, Waa = parameters["Wax"], parameters["Waa"]

    # Push the incoming gradient back through the tanh.
    dtanh = (1 - a_next ** 2) * da_next

    # Gradient w.r.t. the input weights and the input itself.
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    # Gradient w.r.t. the recurrent weights and the previous state.
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    # Gradient w.r.t. the bias: sum across the batch.
    dba = np.sum(dtanh, axis=1, keepdims=True)

    return {"dxt": dxt, "da_prev": da_prev,
            "dWax": dWax, "dWaa": dWaa, "dba": dba}


def rnn_backward(da, caches):
    """Backpropagate through the WHOLE sequence, last step to first.

    `da` holds, for every time step, the gradient arriving from that
    step's own output. But each hidden state also affects the NEXT
    step, so the real gradient at step t is the sum of:
        (a) the gradient from step t's output  -> da[:, :, t]
        (b) the gradient flowing back from step t+1 -> da_prevt

    Because all time steps share ONE set of weights, we ADD each
    step's weight-gradient into a single running total. That
    accumulation is backpropagation through time in one sentence.
    """
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]

    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # Running totals for the shared weights start at zero...
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))  # gradient coming from the future step

    # Walk BACKWARD through time: T_x-1, ..., 2, 1, 0.
    for t in reversed(range(T_x)):
        # Total gradient at this step = its own output's gradient
        # PLUS whatever flowed back from the step after it.
        da_t = da[:, :, t] + da_prevt
        gradients = rnn_cell_backward(da_t, caches[t])

        dxt = gradients["dxt"]
        da_prevt = gradients["da_prev"]   # hand this to the earlier step

        dx[:, :, t] = dxt
        dWax += gradients["dWax"]  # ACCUMULATE shared-weight gradients
        dWaa += gradients["dWaa"]
        dba += gradients["dba"]

    # After the loop, da_prevt is the gradient w.r.t. the very first
    # previous state -- the initial memory a0.
    da0 = da_prevt

    return {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}


def main():
    """A small, concrete run so every shape is visible.

    We invent tiny dimensions and random weights, run the forward
    pass over a 4-step sequence, invent a pretend gradient on the
    hidden states, and run the backward pass. The point is to watch
    the gradient shapes line up with the weight shapes.
    """
    np.random.seed(1)

    n_x, n_a, n_y = 3, 5, 2   # vocab, hidden, output sizes
    m, T_x = 10, 4            # batch size, sequence length

    # Random toy data and starting memory.
    x = np.random.randn(n_x, m, T_x)
    a0 = np.random.randn(n_a, m)

    # Random toy weights (in real training these are LEARNED).
    parameters = {
        "Wax": np.random.randn(n_a, n_x),
        "Waa": np.random.randn(n_a, n_a),
        "Wya": np.random.randn(n_y, n_a),
        "ba": np.random.randn(n_a, 1),
        "by": np.random.randn(n_y, 1),
    }

    # --- Forward over the whole sequence ---
    a, y_pred, caches = rnn_forward(x, a0, parameters)
    print("=== Forward pass over a sequence ===\n")
    print("Sequence length T_x        :", T_x)
    print("Hidden states a   shape    :", a.shape, "(n_a, m, T_x)")
    print("Predictions  y_pred shape  :", y_pred.shape, "(n_y, m, T_x)")
    print("Each prediction column sums to 1.0? ",
          np.allclose(np.sum(y_pred, axis=0), 1.0))

    # --- Backward over the whole sequence ---
    # We don't have a real loss here, so we invent a pretend upstream
    # gradient on the hidden states just to exercise the math.
    da = np.random.randn(n_a, m, T_x)
    gradients = rnn_backward(da, caches)

    print("\n=== Backward pass (backprop through time) ===\n")
    print("dWax shape :", gradients["dWax"].shape,
          " matches Wax", parameters["Wax"].shape)
    print("dWaa shape :", gradients["dWaa"].shape,
          " matches Waa", parameters["Waa"].shape)
    print("dba  shape :", gradients["dba"].shape,
          "  matches ba ", parameters["ba"].shape)
    print("dx   shape :", gradients["dx"].shape, "(n_x, m, T_x)")
    print("da0  shape :", gradients["da0"].shape, "(n_a, m)")
    print("\nEvery gradient has the SAME shape as the thing it updates.")
    print("dWax, dWaa, and dba are SUMS over all", T_x,
          "time steps -- that is backprop through time.")


if __name__ == "__main__":
    main()
