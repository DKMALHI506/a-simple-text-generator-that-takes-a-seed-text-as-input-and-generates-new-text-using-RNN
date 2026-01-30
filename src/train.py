# train.py
# This file trains the RNN model using LSTM

import tensorflow as tf
import numpy as np
import os

# 1. Read the text file
text = open("../data/input.txt", "r", encoding="utf-8").read()

# 2. Create a list of unique characters
vocab = sorted(set(text))

# 3. Create character to number mapping
char2idx = {}
idx2char = {}

for i, ch in enumerate(vocab):
    char2idx[ch] = i
    idx2char[i] = ch

# 4. Convert entire text to numbers
text_as_int = [char2idx[c] for c in text]

# 5. Create input and output sequences
seq_length = 50
input_text = []
target_text = []

for i in range(len(text_as_int) - seq_length):
    input_text.append(text_as_int[i:i+seq_length])
    target_text.append(text_as_int[i+1:i+seq_length+1])

input_text = np.array(input_text)
target_text = np.array(target_text)

# 6. Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(vocab), 64),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dense(len(vocab))
])

# 7. Compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

# 8. Train the model
model.fit(input_text, target_text, epochs=10)

# 9. Save the trained model
model.save("../saved_model/text_generator.h5")

print("Training complete and model saved.")
