import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Load dataset
data = pd.read_csv("dataset/spam.csv")

messages = data['message']
labels = data['label']

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(messages)

sequences = tokenizer.texts_to_sequences(messages)
X = pad_sequences(sequences, maxlen=100)
y = labels

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build LSTM model
model = Sequential()
model.add(Embedding(5000, 64, input_length=100))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model
model.save("model/lstm_model.h5")

# Save tokenizer
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model training complete!")