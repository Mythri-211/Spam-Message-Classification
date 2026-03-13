import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("model/lstm_model.h5")

# Load tokenizer
with open("model/tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

def predict_spam(message):

    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq,maxlen=100)

    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        label = "Spam"
    else:
        label = "Ham"

    return label, prediction