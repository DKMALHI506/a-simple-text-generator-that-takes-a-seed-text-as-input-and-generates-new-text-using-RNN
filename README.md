# A-simple-text-generator-that-takes-a-seed-text-as-input-and-generates-new-text-using-RNN



# Step 1: Data Preparation
Start by gathering a dataset of text that you'll use to train your model. You can use books, articles, or any other text source. Preprocess the text by tokenizing it into words or characters, and create a mapping from words (or characters) to numerical values.

# Step 2: Model Architecture
Build an RNN model. In this example, let's use a simple LSTM (Long Short-Term Memory) network, a type of RNN.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

vocab_size = len(vocab)  # Replace with the actual vocabulary size
embedding_dim = 256
rnn_units = 1024

model = Sequential([
    Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    LSTM(rnn_units, return_sequences=True, stateful=True),
    Dense(vocab_size)
])

# Step 3: Loss and Optimization
Define the loss function and optimizer for training your model. Since this is a text generation task, you can use the SparseCategoricalCrossentropy loss.
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Step 4: Training the Model
Train your model on the prepared dataset. During training, the model will learn the patterns and relationships between words in the text.

# Replace with your training data and labels
train_data = ...
train_labels = ...

model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

# Step 5: Generating Text

Now that you've trained your model, you can use it to generate text. Start with a seed text and iteratively predict the next word based on the previous words.

def generate_text(model, start_string, num_generate=1000):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    
    text_generated = []

    model.reset_states()
    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


# Step 6: Generating Text
Finally, use the generate_text function to generate new text based on a seed string.

seed_text = "Once upon a time"
generated_text = generate_text(model, start_string=seed_text, num_generate=200)
print(generated_text)
