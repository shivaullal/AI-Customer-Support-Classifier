from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib


# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------

app = FastAPI(
    title="AI Customer Support Ticket Classifier"
)


# ---------------------------------------------------------
# Templates
# ---------------------------------------------------------

templates = Jinja2Templates(
    directory="templates"
)


# ---------------------------------------------------------
# Static Files
# ---------------------------------------------------------
# Your CSS file is:
# static/style.css
#
# Therefore HTML should use:
# {{ url_for('static', path='style.css') }}

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ---------------------------------------------------------
# Load Trained ML Model
# ---------------------------------------------------------

model = joblib.load(
    "model/ticket_classifier.pkl"
)


# ---------------------------------------------------------
# Home Page
# ---------------------------------------------------------

@app.get("/")
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )


# ---------------------------------------------------------
# Classify Ticket
# ---------------------------------------------------------

@app.post("/classify")
def classify(
    request: Request,
    ticket: str = Form(...)
):

    # Get message
    message = ticket.strip()


    # -----------------------------------------------------
    # Predict Category
    # -----------------------------------------------------

    prediction = model.predict(
        [message]
    )[0]


    # -----------------------------------------------------
    # Calculate Confidence
    # -----------------------------------------------------

    probabilities = model.predict_proba(
        [message]
    )[0]

    confidence = max(probabilities) * 100


    # -----------------------------------------------------
    # Determine Priority & Department
    # -----------------------------------------------------

    if prediction == "Payment":

        priority = "High"
        department = "Billing"

    elif prediction == "Account":

        priority = "Medium"
        department = "Account Support"

    elif prediction == "Delivery":

        priority = "High"
        department = "Delivery Support"

    else:

        priority = "Medium"
        department = "Technical Support"


    # -----------------------------------------------------
    # Create Result
    # -----------------------------------------------------

    result = {

        "message": message,

        "category": prediction,

        "confidence": round(
            confidence,
            2
        ),

        "priority": priority,

        "department": department
    }


    # -----------------------------------------------------
    # Return Result
    # -----------------------------------------------------

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": result,
            "ticket": message
        }
    )