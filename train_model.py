from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os


# Training data
tickets = [
    # Payment
    "Money was deducted from my account",
    "I was charged twice",
    "My payment failed",
    "My card was charged",
    "Payment was deducted but order was cancelled",
    "I have a payment problem",
    "I was charged for the same order twice",

    # Account
    "I cannot login",
    "I forgot my password",
    "I cannot access my account",
    "Please help me reset my password",
    "My account is locked",
    "I cannot sign in",

    # Delivery
    "My package has not arrived",
    "Where is my order?",
    "My delivery is late",
    "The package was not delivered",
    "I have not received my order",
    "My delivery is delayed",

    # Technical
    "The application is crashing",
    "The website is not working",
    "The app keeps crashing",
    "I found a technical problem",
    "The page is not loading",
    "The website gives me an error"
]


categories = [
    "Payment",
    "Payment",
    "Payment",
    "Payment",
    "Payment",
    "Payment",
    "Payment",

    "Account",
    "Account",
    "Account",
    "Account",
    "Account",
    "Account",

    "Delivery",
    "Delivery",
    "Delivery",
    "Delivery",
    "Delivery",
    "Delivery",

    "Technical",
    "Technical",
    "Technical",
    "Technical",
    "Technical",
    "Technical"
]


# Create ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )
    ),

    (
        "classifier",
        LogisticRegression()
    )
])


# Train model
model.fit(tickets, categories)


# Create model directory
os.makedirs("model", exist_ok=True)


# Save trained model
joblib.dump(
    model,
    "model/ticket_classifier.pkl"
)


print("--------------------------------")
print("MODEL TRAINED SUCCESSFULLY")
print("--------------------------------")
print("Model saved at:")
print("model/ticket_classifier.pkl")