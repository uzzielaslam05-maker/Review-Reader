# Review Reader 🎬

A Simple RNN sentiment classifier trained on 50,000 IMDb movie reviews — with a live, interactive web interface where you can type any review and get an instant Positive/Negative prediction.

**🔗 Live demo:** [uzzielcedric.pythonanywhere.com](https://uzzielcedric.pythonanywhere.com)

---

## What this is

A complete pipeline from raw text to a deployed model:

1. **Preprocessing** — HTML/punctuation stripping, tokenization (10k-word vocabulary), sequence padding
2. **Model** — Embedding layer → SimpleRNN → Dense, trained from scratch with masking and regularization to prevent overfitting
3. **Deployment** — the trained model, converted to TensorFlow Lite and served through a lightweight Flask app with a custom-built interface

## Results

Trained on 35,000 reviews, validated on 7,500, tested on 7,500 held-out reviews.

| Metric | Score |
|---|---|
| Test accuracy | **83.4%** |
| Precision (positive) | 0.80 |
| Recall (positive) | 0.90 |
| Precision (negative) | 0.88 |
| Recall (negative) | 0.77 |

## Why a plain SimpleRNN and not LSTM/GRU

This project deliberately uses a vanilla `SimpleRNN` layer rather than the more common LSTM/GRU choice, which surfaced a few real problems worth documenting:

- **Long sequences (200+ tokens) caused the model to not learn at all** — a textbook vanishing-gradient failure. Fixed by shortening sequences to 120 tokens and using a single RNN layer instead of stacking multiple.
- **Short reviews (a handful of words) were misclassified almost uniformly**, because padding tokens diluted the signal from the real words. Fixed with `mask_zero=True` on the Embedding layer and pre-padding (`padding="pre"`) instead of post-padding.
- **TFLite conversion initially required TensorFlow's "Flex ops" delegate** (defeating the point of a lightweight export) because of how Keras implements masked RNNs internally. Fixed by adding `unroll=True` to the `SimpleRNN` layer, which forces a static graph that converts to pure TFLite built-in ops — no Flex delegate, no full TensorFlow dependency at inference time.

## Project structure

```
Review-Reader/
├── app.py                    # Flask app: web interface + inference (PythonAnywhere-ready)
├── imdb_sentiment.tflite     # Trained model, exported to TensorFlow Lite (~1.4MB)
├── tokenizer.json            # Vocabulary/word-index, exported as plain JSON (no TF needed to load)
├── requirements.txt          # flask, numpy, ai-edge-litert
├── imdb_simple_rnn.py        # Full training script: preprocessing → training → evaluation
└── predict_review.py         # Standalone CLI script for testing the model locally
```

## Running it locally

```bash
pip install flask numpy ai-edge-litert
python app.py
```

Then open `http://127.0.0.1:5000`.

## Training it yourself

Requires the [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (`IMDB_Dataset.csv`).

```bash
pip install tensorflow pandas scikit-learn matplotlib
python imdb_simple_rnn.py
```

This retrains the model from scratch, saves a `.keras` model, and prints evaluation metrics + a training curve plot. To redeploy after retraining, re-export to TFLite:

```python
import tensorflow as tf
model = tf.keras.models.load_model("simple_rnn_imdb.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
open("imdb_sentiment.tflite", "wb").write(converter.convert())
```

## Tech stack

- **Model:** TensorFlow/Keras (training) → TensorFlow Lite (deployment)
- **Backend:** Flask
- **Inference runtime:** [ai-edge-litert](https://pypi.org/project/ai-edge-litert/) (~48MB, vs. ~1.9GB for full TensorFlow)
- **Frontend:** vanilla HTML/CSS/JS, no build step
- **Hosting:** [PythonAnywhere](https://www.pythonanywhere.com) free tier

## Known limitations

- Negation and sarcasm ("not bad", "yeah, right, amazing...") can still trip up the model — a known limitation of plain RNNs without attention.
- Vocabulary is capped at the 10,000 most frequent words seen in training; rarer words map to an out-of-vocabulary token.
