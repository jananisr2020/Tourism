"""Streamlit app: predict whether a customer will buy the Wellness Tourism Package."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧘", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load(HERE / "best_model.joblib")
    meta = json.loads((HERE / "model_meta.json").read_text())
    return model, meta


model, meta = load_model()
threshold = meta["threshold"]

st.title("🧘 Wellness Tourism Package: Purchase Predictor")
st.caption("Visit with Us · enter customer details to estimate the chance they buy the new package.")

with st.sidebar:
    st.subheader("Model")
    for k, v in meta["test_metrics"].items():
        st.metric(k.replace("test_", "").replace("_", " ").title(), v)
    st.caption(f"Decision threshold: {threshold}")

st.subheader("Customer details")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", 18, 80, 36)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly income", 1000, 150000, 22000, step=500)
with c2:
    city = st.selectbox("City tier", [1, 2, 3])
    persons = st.number_input("Number of people visiting", 1, 10, 3)
    children = st.number_input("Children under 5 visiting", 0, 5, 1)
    star = st.selectbox("Preferred property star", [3.0, 4.0, 5.0])
    trips = st.number_input("Trips per year", 0, 30, 3)
    passport = st.selectbox("Holds a valid passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns a car?", ["No", "Yes"])
with c3:
    contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration = st.number_input("Pitch duration (minutes)", 1, 130, 15)
    followups = st.number_input("Follow-ups after pitch", 0, 10, 4)
    satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)

row = pd.DataFrame([{
    "Age": age, "TypeofContact": contact, "CityTier": city, "DurationOfPitch": duration,
    "Occupation": occupation, "Gender": gender, "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups, "ProductPitched": product, "PreferredPropertyStar": star,
    "MaritalStatus": marital, "NumberOfTrips": trips, "Passport": int(passport == "Yes"),
    "PitchSatisfactionScore": satisfaction, "OwnCar": int(own_car == "Yes"),
    "NumberOfChildrenVisiting": children, "Designation": designation, "MonthlyIncome": income,
}])

if st.button("Predict", type="primary"):
    prob = float(model.predict_proba(row)[0, 1])
    left, right = st.columns([1, 2])
    left.metric("Purchase probability", f"{prob:.1%}")
    right.progress(min(prob, 1.0))
    if prob >= threshold:
        st.success("✅ Likely to purchase — prioritise this customer for the campaign.")
    else:
        st.warning("⚠️ Unlikely to purchase — lower priority for this campaign.")
    with st.expander("Input sent to the model"):
        st.dataframe(row, hide_index=True)
