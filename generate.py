# generate.py
# This file generates new text using the trained model

import tensorflow as tf
import numpy as np

# Load training text again (to rebuild vocab)
text = open("../data/input.txt", "r", encoding="utf-8").read()
vocab = sorted(set(text))

char2idx = {ch:i for i,ch in enumerate(vocab)}
idx2char = {i:ch for i,ch in enumerate(vocab)}

# Load trained model
model = tf.keras.models.load_model("../saved_model/text_generator.h5")

def generate_text(start_text, num_chars=200):
    input_chars = [char2idx[c] for c in start_text]
    input_chars = tf.expand_dims(input_chars, 0)

    generated_text = start_text

    for _ in range(num_chars):
        predictions = model(input_chars)
        last_prediction = predictions[:, -1, :]
        predicted_id = tf.random.categorical(last_prediction, 1)[0][0].numpy()

        generated_text += idx2char[predicted_id]
        input_chars = tf.expand_dims([predicted_id], 0)

    return generated_text

print(generate_text("Once upon a time"))
