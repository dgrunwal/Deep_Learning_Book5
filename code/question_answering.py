"""
question_answering.py
==================================================================
Companion script for Chapter 11 of "Deep Learning for Beginners:
Sequence Models" (AI Foundations Series, Book 5).

WHAT THIS SCRIPT DOES
-----------------------------------------------------------------
It answers questions about a paragraph, using a pretrained
transformer - EXTRACTIVE question answering:

    context : "The kitchen is east of the hallway. The garden is
               south of the bedroom."
    question: "What is east of the hallway?"
    answer  : "the kitchen"

"Extractive" means the answer is always a SPAN copied straight out
of the context - a start word and an end word - not text the model
writes freely. So the model's whole job is to point at two places:
where the answer BEGINS and where it ENDS.

It mirrors the Course-5 Week-4 question-answering lab (DistilBERT).

HOW THE MODEL POINTS AT A SPAN
-----------------------------------------------------------------
You feed the transformer the QUESTION and the CONTEXT together, and
for every token it outputs TWO scores: how likely this token is the
START of the answer, and how likely it is the END. Take the token
with the highest start score and the token with the highest end
score, and the words between them are the answer. Two softmaxes over
the tokens - that is the entire trick.
==================================================================
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# =================================================================
# The real HuggingFace question-answering workflow
# =================================================================
MODEL_NAME = "distilbert-base-cased-distilled-squad"


def run_real_qa_torch(examples):
    """Answer questions with a pretrained DistilBERT QA model (PyTorch).

    This is the code you would actually run. It needs `transformers` and
    `torch`, and downloads a small SQuAD-tuned model the first time.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.eval()

    for question, context in examples:
        # Feed QUESTION and CONTEXT together; the tokenizer joins them
        # with a [SEP] marker so the model knows which part is which.
        inputs = tokenizer(question, context, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # Two score vectors, one per token: start- and end-likelihood.
        start = int(torch.argmax(outputs.start_logits, dim=1)[0])
        end = int(torch.argmax(outputs.end_logits, dim=1)[0]) + 1

        ids = inputs["input_ids"][0][start:end]
        answer = tokenizer.decode(ids)
        print(f"   Q: {question}")
        print(f"   A: {answer}\n")


def run_real_qa_tf(examples):
    """Same QA, TensorFlow version (used if torch is absent)."""
    from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
    import tensorflow as tf

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TFAutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    for question, context in examples:
        inputs = tokenizer(question, context, return_tensors="tf")
        outputs = model(inputs)
        start = int(tf.argmax(outputs.start_logits, axis=1)[0])
        end = int(tf.argmax(outputs.end_logits, axis=1)[0]) + 1
        ids = inputs["input_ids"][0][start:end]
        answer = tokenizer.decode(ids)
        print(f"   Q: {question}")
        print(f"   A: {answer}\n")


# =================================================================
# Fallback: show the MECHANISM without any downloads
# =================================================================
def run_concept_demo(examples):
    """Show start/end span extraction with hand-made scores.

    We fake the two score vectors a real model would output, so you can
    see exactly how start + end argmax turns into an answer span.
    """
    print("(No usable transformers backend - showing the QA MECHANISM instead.)\n")

    question, context = examples[0]
    tokens = context.replace(".", " .").split()

    print(f"   context : {context}")
    print(f"   question: {question}\n")
    print("A real model outputs a START score and an END score for every token.")
    print("It then picks the best start and the best end; the words between")
    print("them are the answer. For 'What is east of the hallway?' the highest")
    print("start score lands on 'The' and the highest end score on 'kitchen':\n")

    # Pretend the model chose these two token positions.
    try:
        start = tokens.index("The")
        end = tokens.index("kitchen")
    except ValueError:
        start, end = 0, 1
    answer = " ".join(tokens[start:end + 1])
    for i, tok in enumerate(tokens):
        marker = ""
        if i == start:
            marker = "  <- START (highest start score)"
        elif i == end:
            marker = "  <- END (highest end score)"
        print(f"   token {i:2d}: {tok:10s}{marker}")
    print(f"\n   Extracted answer span: \"{answer}\"")


def _pick_backend():
    """Return 'torch', 'tf', or None depending on what is installed.

    `transformers` needs a backend (PyTorch or TensorFlow) to load models.
    We prefer PyTorch because it installs cleanly on the newest Pythons.
    """
    try:
        import transformers  # noqa: F401
    except ImportError:
        return None
    try:
        import torch  # noqa: F401
        return "torch"
    except ImportError:
        pass
    try:
        import tensorflow  # noqa: F401
        return "tf"
    except ImportError:
        return None


def main():
    examples = [
        ("What is east of the hallway?",
         "The kitchen is east of the hallway. The garden is south of the bedroom."),
        ("Where is the garden?",
         "The kitchen is east of the hallway. The garden is south of the bedroom."),
        ("Who founded the company?",
         "The company was founded by Ada Lovelace in London in 1850."),
    ]

    backend = _pick_backend()
    try:
        if backend == "torch":
            print("Answering questions with a pretrained transformer (PyTorch):\n")
            run_real_qa_torch(examples)
        elif backend == "tf":
            print("Answering questions with a pretrained transformer (TensorFlow):\n")
            run_real_qa_tf(examples)
        else:
            raise ImportError(
                "no model backend found (need PyTorch or TensorFlow)")
    except Exception as exc:                                  # noqa: BLE001
        print(f"[note] {type(exc).__name__}: {exc}\n")
        run_concept_demo(examples)
        print("\nTo run the real model, install transformers AND a backend:")
        print("   pip install transformers torch      (PyTorch - recommended)")
        print("   pip install transformers tensorflow (TensorFlow alternative)")

    print("\nDone. Extractive QA = point at the start and end of the answer")
    print("inside the context. The transformer never writes new words; it")
    print("just learns where the answer already sits in the passage.")


if __name__ == "__main__":
    main()
