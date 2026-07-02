"""
ner_finetune.py
==================================================================
Companion script for Chapter 11 of "Deep Learning for Beginners:
Sequence Models" (AI Foundations Series, Book 5).

WHAT THIS SCRIPT DOES
-----------------------------------------------------------------
It fine-tunes a pretrained transformer for NAMED ENTITY RECOGNITION
(NER) - the task of finding and labeling the people, places, and
organizations in a sentence:

    "Barack Obama visited Google in California"
       PER   PER            ORG      LOC

This is the modern NLP workflow in a nutshell: you do NOT build a
transformer from scratch. You download one that was already trained
on huge amounts of text, and you FINE-TUNE it - nudge its weights on
your smaller labeled dataset for a couple of epochs. The whole thing
is a few lines with HuggingFace `transformers` (on a PyTorch or
TensorFlow backend).

It mirrors the Course-5 Week-4 NER lab (DistilBERT for token
classification).

HOW NER IS FRAMED: TOKEN CLASSIFICATION
-----------------------------------------------------------------
NER is just classification done once PER TOKEN. The transformer reads
the sentence and, for every token, predicts a tag. Tags use the "BIO"
scheme: B-PER (beginning of a person), I-PER (inside/continuation of
a person), B-LOC, B-ORG, and O (outside any entity). So "Barack
Obama" becomes B-PER I-PER, and plain words like "visited" become O.

THE ONE TRICKY BIT: SUBWORD ALIGNMENT
-----------------------------------------------------------------
A subword tokenizer may split one word into several tokens
("California" -> "Cali", "##fornia"). But your labels are per-WORD.
So you must ALIGN them: give the first sub-token the word's real
label, and mark the leftover sub-tokens (and the special [CLS]/[SEP]
tokens) with -100, a magic value that tells the loss to ignore them.
That alignment is the heart of the script.
==================================================================
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# The labels we will predict, and integer <-> string maps.
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
tag2id = {t: i for i, t in enumerate(LABELS)}
id2tag = {i: t for t, i in tag2id.items()}


# =================================================================
# The alignment step - the one genuinely fiddly part of NER
# =================================================================
def align_labels_with_tokens(word_labels, word_ids):
    """Spread per-WORD labels onto per-TOKEN labels.

    word_labels -- the label id for each ORIGINAL word
    word_ids    -- for each token the tokenizer produced, which original
                   word it came from (None for special tokens like [CLS])

    Rule (the standard HuggingFace recipe):
      * special token (word_id is None)        -> -100  (ignored in loss)
      * first token of a word                  -> the word's real label
      * later sub-tokens of the same word      -> -100  (ignored)
    -100 is a sentinel: the loss function skips any position labeled -100,
    so leftover sub-tokens never confuse training.
    """
    aligned = []
    previous_word = None
    for wid in word_ids:
        if wid is None:
            aligned.append(-100)
        elif wid != previous_word:
            aligned.append(word_labels[wid])     # first sub-token gets the label
        else:
            aligned.append(-100)                 # continuation sub-token ignored
        previous_word = wid
    return aligned


# =================================================================
# The real HuggingFace fine-tuning workflow
# =================================================================
# A tiny hand-labeled dataset, shared by both backends
# (word lists + per-word BIO tags).
SENTENCES = [
    ["Barack", "Obama", "visited", "Google", "in", "California"],
    ["Angela", "Merkel", "met", "Tim", "Cook", "in", "Berlin"],
    ["Amazon", "opened", "an", "office", "in", "Seattle"],
]
TAGS = [
    ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC"],
    ["B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-LOC"],
    ["B-ORG", "O", "O", "O", "O", "B-LOC"],
]
TEST_SENTENCE = ["Sundar", "Pichai", "leads", "Google", "in", "Mountain", "View"]
MODEL_NAME = "distilbert-base-uncased"


def _print_predictions(test_words, preds, word_ids):
    """Print one predicted tag per original word (first sub-token only)."""
    print("\nPredicted entities:")
    seen = set()
    for tok_idx, wid in enumerate(word_ids):
        if wid is not None and wid not in seen:
            seen.add(wid)
            print(f"   {test_words[wid]:12s} -> {id2tag[int(preds[tok_idx])]}")


def run_real_finetune_torch():
    """Fine-tune DistilBERT for token classification with PyTorch.

    This is the code you would actually run. It needs `transformers` and
    `torch`, and downloads a small pretrained model the first time.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize (splitting into subwords) and align the labels.
    enc = tokenizer(SENTENCES, is_split_into_words=True, truncation=True,
                    padding=True, return_tensors="pt")
    all_labels = []
    for i, sent_tags in enumerate(TAGS):
        word_ids = enc.word_ids(batch_index=i)
        label_ids = [tag2id[t] for t in sent_tags]
        all_labels.append(align_labels_with_tokens(label_ids, word_ids))
    labels = torch.tensor(all_labels)

    # Load the pretrained transformer with a fresh token-classification head.
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=id2tag, label2id=tag2id)

    # A plain manual training loop - the clearest way to see fine-tuning.
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    print("Fine-tuning DistilBERT for NER with PyTorch (a few epochs) ...")
    for epoch in range(8):
        optimizer.zero_grad()
        # Passing labels makes the model return the loss directly.
        out = model(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    labels=labels)
        out.loss.backward()          # backprop
        optimizer.step()             # nudge the weights
        print(f"   epoch {epoch + 1}/8   loss = {out.loss.item():.4f}")

    # Predict on a new sentence.
    model.eval()
    tenc = tokenizer([TEST_SENTENCE], is_split_into_words=True,
                     return_tensors="pt")
    with torch.no_grad():
        logits = model(**tenc).logits
    preds = torch.argmax(logits, dim=-1)[0].tolist()
    _print_predictions(TEST_SENTENCE, preds, tenc.word_ids(batch_index=0))


def run_real_finetune_tf():
    """Same fine-tuning, TensorFlow/Keras version (used if torch is absent)."""
    from transformers import AutoTokenizer, TFAutoModelForTokenClassification
    import tensorflow as tf

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = tokenizer(SENTENCES, is_split_into_words=True, truncation=True,
                    padding=True, return_tensors="tf")
    all_labels = []
    for i, sent_tags in enumerate(TAGS):
        word_ids = enc.word_ids(batch_index=i)
        label_ids = [tag2id[t] for t in sent_tags]
        all_labels.append(align_labels_with_tokens(label_ids, word_ids))
    labels = tf.constant(all_labels)

    model = TFAutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=id2tag, label2id=tag2id)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5))
    print("Fine-tuning DistilBERT for NER with TensorFlow (a few epochs) ...")
    model.fit(dict(enc), labels, epochs=8, batch_size=3, verbose=2)

    tenc = tokenizer([TEST_SENTENCE], is_split_into_words=True,
                     return_tensors="tf")
    logits = model(dict(tenc)).logits
    preds = tf.argmax(logits, axis=-1)[0].numpy()
    _print_predictions(TEST_SENTENCE, preds, tenc.word_ids(batch_index=0))


# =================================================================
# Fallback: show the MECHANISM without any downloads
# =================================================================
def run_concept_demo():
    """Illustrate subword tokenization + BIO labeling + alignment.

    No transformer, no training - just the data transformations that
    make NER work, so the idea is visible even offline.
    """
    print("(No usable transformers backend - showing the NER MECHANISM instead.)\n")

    sentence = ["Barack", "Obama", "visited", "Google", "in", "California"]
    word_tags = ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC"]

    # A pretend subword tokenizer that splits long words with "##".
    def fake_subword(word):
        if len(word) > 6:
            return [word[:4], "##" + word[4:]]
        return [word]

    print("1. Subword tokenization (long words split into pieces):")
    tokens, word_ids = ["[CLS]"], [None]
    for wid, w in enumerate(sentence):
        for piece in fake_subword(w):
            tokens.append(piece)
            word_ids.append(wid)
    tokens.append("[SEP]"); word_ids.append(None)
    print("   " + " ".join(tokens))

    print("\n2. Align per-word labels onto per-token labels:")
    label_ids = [tag2id[t] for t in word_tags]
    aligned = align_labels_with_tokens(label_ids, word_ids)
    for tok, a in zip(tokens, aligned):
        shown = id2tag[a] if a != -100 else "(ignored, -100)"
        print(f"   {tok:14s} -> {shown}")

    print("\nThat -100 tells the loss to skip special tokens and the leftover")
    print("sub-token pieces, so only the FIRST piece of each word is scored.")
    print("A real transformer then learns to predict these tags from context.")


def _pick_backend():
    """Return 'torch', 'tf', or None depending on what is installed.

    `transformers` needs a backend (PyTorch or TensorFlow) to load models;
    on its own it only provides tokenizers. We prefer PyTorch because it
    installs cleanly on the newest Python versions.
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
    backend = _pick_backend()
    try:
        if backend == "torch":
            run_real_finetune_torch()
        elif backend == "tf":
            run_real_finetune_tf()
        else:
            # transformers missing, or present but with no model backend.
            raise ImportError(
                "no model backend found (need PyTorch or TensorFlow)")
    except Exception as exc:                                  # noqa: BLE001
        print(f"[note] {type(exc).__name__}: {exc}\n")
        run_concept_demo()
        print("\nTo run the real fine-tuning, install transformers AND a backend:")
        print("   pip install transformers torch      (PyTorch - recommended)")
        print("   pip install transformers tensorflow (TensorFlow alternative)")

    print("\nDone. NER = classify every token. The only fiddly part is aligning")
    print("word-level labels to subword tokens; a pretrained transformer does")
    print("the rest, fine-tuned in just a few lines and a few epochs.")


if __name__ == "__main__":
    main()
