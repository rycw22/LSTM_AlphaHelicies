from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# load trained model
model = tf.keras.models.load_model("helix_model.h5")

app = FastAPI()

# amino acid lookup (must match training encoding)
amino_acids = list("acdefghiklmnpqrstvwy")
amino_index = {aa: i + 1 for i, aa in enumerate(amino_acids)}

class SequenceInput(BaseModel):
    sequence: str   # protein sequence (string of amino acids)

@app.post("/predict")
def predict_helices(input_data: SequenceInput):
    seq = input_data.sequence.lower()
    X = [amino_index.get(aa, 0) for aa in seq]  # encode
    X_onehot = np.eye(len(amino_acids) + 1)[X]
    X_onehot = np.expand_dims(X_onehot, axis=0)  # batch dim

    pred_probs = model.predict(X_onehot)
    helix_index = 1  # "h" class
    probs = pred_probs[0, :, helix_index].tolist()

    return {"helix_probabilities": probs}
