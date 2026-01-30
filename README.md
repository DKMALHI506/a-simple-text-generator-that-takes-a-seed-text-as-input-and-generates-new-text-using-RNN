# Simple Text Generator Using RNN

This project is a basic text generator built using
Recurrent Neural Networks (LSTM) in TensorFlow.

The model learns from a text file and then generates
new text based on a starting sentence.

## Files Used
- input.txt – training text
- train.py – trains the model
- generate.py – generates new text

## How It Works
1. Text is converted into characters
2. Characters are mapped to numbers
3. LSTM learns character patterns
4. Model predicts next character

## How to Run

### Step 1: Install libraries
pip install tensorflow numpy

### Step 2: Train the model
python src/train.py

### Step 3: Generate text
python src/generate.py

## Example Output
Once upon a time there was a small village...

