import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
Sequential = tf.keras.models.Sequential
Bidirectional = tf.keras.layers.Bidirectional
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
TimeDistributed = tf.keras.layers.TimeDistributed
Masking = tf.keras.layers.Masking
Adam = tf.keras.optimizers.Adam
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
auc = tf.keras.metrics.AUC()

'''
configuration
'''
max_len = 400
batch = 32
training = "/Users/ryancw/Downloads/Protein Project/Protein_Project/protein-secondary-structure.train"


# means of one hot encoding
amino_acids = list("acdefghiklmnpqrstvwy")
amino_index = {aa: i + 1 for i, aa in enumerate(amino_acids)}
unk_idx = len(amino_index) + 1  
vocab_size = unk_idx + 1        
structure_codes = ["_", "h", "e"]
structure_index = {ss: i for i, ss in enumerate(structure_codes)}

helix_labels = set(['h'])

'''
parsing through data (and also data preparation)
'''
sequences = []
labels = []

current_seq = []
current_lbl = []

with open(training, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        # Treat both "<>" and "end" as sequence delimiters
        if line in {"<>", "end"}:
            if current_seq:
                sequences.append(current_seq)
                labels.append(current_lbl)
                current_seq = []
                current_lbl = []
        else:
            parts = line.split()
            if len(parts) != 2:
                continue  # skip malformed lines
            aa, ss = parts
            current_seq.append(amino_index.get(aa.lower(), 0))  # lowercase to match amino_acids list
            current_lbl.append(structure_index.get(ss.lower(), 0))  # 0 for coil or unknown

# Add last sequence if not empty
if current_seq:
    sequences.append(current_seq)
    labels.append(current_lbl)

# Pad sequences
max_len = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_len, padding="post")
y = pad_sequences(labels, maxlen=max_len, padding="post")

# One-hot encode for model input
X_onehot = np.eye(len(amino_acids) + 1)[X]  # +1 because we have padding=0
y_onehot = np.eye(len(structure_codes))[y]

num_features = X_onehot.shape[2]
num_classes = y_onehot.shape[2]

model = Sequential([
    Masking(mask_value=0.0, input_shape=(None, num_features)),  # mask padding
    Bidirectional(LSTM(64, return_sequences=True)),
    TimeDistributed(Dense(num_classes, activation='softmax'))
])

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'), 
             tf.keras.metrics.Recall(name='recall'), 
             tf.keras.metrics.AUC(name='auc')]
)

# Train
history = model.fit(
    X_onehot, y_onehot,
    validation_split=0.1,
    epochs=10,
    batch_size=32
)

model.save("helix_model.h5")

# Example: predict helix probabilities
pred_probs = model.predict(X_onehot)

# Get per-residue helix probability (assuming 'h' or 'H' is class index 1 or similar)
helix_index = 1  # adjust based on your ss_to_idx mapping
helix_probs = pred_probs[:, :, helix_index]
'''
- print out where the alpha helicies are - and add the other means of scoring
- how do we figure out accuracy for the model
    - look into how the accuracy is calculated and if we need to change the method of calculation 
    - what does 85% accuracy mean?

'''
loss, accuracy, precision, rec, auc, f1_score = model.evaluate(X_onehot, y_onehot, verbose=0) # added f1_score

# add test set :( 80 - 10 - 10 ratio
# add f1 score

def get_helices(probs, threshold=0.8, window_size=4):
    helices = []
    in_helix = False
    start = None
    
    for i, p in enumerate(probs):
        if p >= threshold:
            if not in_helix:
                in_helix = True
                start = i
        else:
            if in_helix:
                end = i - 1
                if (end - start + 1) >= window_size:
                    helices.append((start, end))
                in_helix = False
    # handle helix reaching end of sequence
    if in_helix:
        end = len(probs) - 1
        if (end - start + 1) >= window_size:
            helices.append((start, end))
    return helices



for seq_idx in range(3):
    helices = get_helices(helix_probs[seq_idx], threshold=0.8, window_size=4)
    print(f"Sequence {seq_idx+1}: helices {helices}")


# save the helicies into a csv file
'''
how to add information: 
1) imporve the method
- How did you implement and classify it
- Create documentation on WHY!!
- HOW are you adding improvements to it?
- Using basic LSTM from Tensorflow - generally used for text prediction (has a good relationship prediction) 
that helps with  relationship
- 

2) '''
