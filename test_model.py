import joblib


model = joblib.load(
    "model/ticket_classifier.pkl"
)


messages = [
    "My money was deducted",
    "I forgot my password",
    "My package hasn't arrived",
    "The application keeps crashing"
]


for message in messages:

    prediction = model.predict([message])[0]

    print()
    print("Message:", message)
    print("Prediction:", prediction)