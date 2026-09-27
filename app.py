import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# 1. Load the saved models (the .sav file must sit next to app.py)
# ---------------------------------------------------------------
@st.cache_resource
def load_bundle():
    return joblib.load("hotel_models.sav")

bundle = load_bundle()
log_model = bundle["log_model"]
lin_model = bundle["lin_model"]
scaler_clf = bundle["scaler_clf"]
scaler_reg = bundle["scaler_reg"]
clf_columns = bundle["clf_columns"]
reg_columns = bundle["reg_columns"]
cat_options = bundle["cat_options"]


def prepare(raw: dict, columns: list, scaler) -> np.ndarray:
    """Turn one booking (plain values) into the exact scaled format the model was trained on."""
    row = pd.DataFrame([raw])
    row = pd.get_dummies(row)                               # one-hot encode text columns
    row = row.reindex(columns=columns, fill_value=0)        # same columns, same order as training
    row = row.astype(float)
    return scaler.transform(row)


def pick(label, col, default=None):
    """Dropdown for a categorical column, using the values seen in training."""
    options = cat_options[col]
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index)


# ---------------------------------------------------------------
# 2. Page layout and inputs
# ---------------------------------------------------------------
st.set_page_config(page_title="Hotel Booking Predictor", page_icon="🏨")
st.title("🏨 Hotel Booking Predictor")
st.write(
    "Enter a booking's details, then predict whether it will be **cancelled** "
    "(Logistic Regression) or what its **average daily rate (ADR)** will be (Linear Regression)."
)

st.subheader("Booking details")
c1, c2 = st.columns(2)

with c1:
    hotel = pick("Hotel", "hotel", "Resort Hotel")
    arrival = st.date_input("Arrival date", datetime.date(2016, 7, 1))
    lead_time = st.number_input("Lead time (days between booking and arrival)", 0, 800, 80)
    weekend_nights = st.number_input("Weekend nights", 0, 20, 1)
    week_nights = st.number_input("Week nights", 0, 50, 3)
    adults = st.number_input("Adults", 0, 10, 2)
    children = st.number_input("Children", 0, 10, 0)
    babies = st.number_input("Babies", 0, 10, 0)
    meal = pick("Meal plan", "meal", "BB")
    country = pick("Country (ISO code)", "country", "PRT")
    market_segment = pick("Market segment", "market_segment", "Online TA")
    distribution_channel = pick("Distribution channel", "distribution_channel", "TA/TO")

with c2:
    is_repeated_guest = st.selectbox("Repeated guest?", ["No", "Yes"]) == "Yes"
    previous_cancellations = st.number_input("Previous cancellations", 0, 30, 0)
    previous_not_canceled = st.number_input("Previous bookings not cancelled", 0, 80, 0)
    reserved_room = pick("Reserved room type", "reserved_room_type", "A")
    assigned_room = pick("Assigned room type", "assigned_room_type", "A")
    booking_changes = st.number_input("Booking changes", 0, 25, 0)
    deposit_type = pick("Deposit type", "deposit_type", "No Deposit")
    waiting_days = st.number_input("Days in waiting list", 0, 400, 0)
    customer_type = pick("Customer type", "customer_type", "Transient")
    parking = st.number_input("Car parking spaces required", 0, 8, 0)
    special_requests = st.number_input("Special requests", 0, 5, 0)

# Same column names the notebook used (before get_dummies)
booking = {
    "hotel": hotel,
    "lead_time": lead_time,
    "arrival_date_year": arrival.year,
    "arrival_date_month": arrival.strftime("%B"),          # e.g. "July"
    "arrival_date_week_number": arrival.isocalendar()[1],
    "arrival_date_day_of_month": arrival.day,
    "stays_in_weekend_nights": weekend_nights,
    "stays_in_week_nights": week_nights,
    "adults": adults,
    "children": children,
    "babies": babies,
    "meal": meal,
    "country": country,
    "market_segment": market_segment,
    "distribution_channel": distribution_channel,
    "is_repeated_guest": int(is_repeated_guest),
    "previous_cancellations": previous_cancellations,
    "previous_bookings_not_canceled": previous_not_canceled,
    "reserved_room_type": reserved_room,
    "assigned_room_type": assigned_room,
    "booking_changes": booking_changes,
    "deposit_type": deposit_type,
    "days_in_waiting_list": waiting_days,
    "customer_type": customer_type,
    "required_car_parking_spaces": parking,
    "total_of_special_requests": special_requests,
}

# ---------------------------------------------------------------
# 3. Predictions
# ---------------------------------------------------------------
tab1, tab2 = st.tabs(["❌ Will it be cancelled?", "💰 Predict daily rate (ADR)"])

with tab1:
    # The cancellation model also used ADR as an input
    adr = st.number_input("Average daily rate (ADR) of the booking", 0.0, 1000.0, 100.0)
    if st.button("Predict cancellation"):
        X = prepare({**booking, "adr": adr}, clf_columns, scaler_clf)
        prob = log_model.predict_proba(X)[0][1]
        if prob >= 0.5:
            st.error(f"Likely to be CANCELLED — probability {prob:.1%}")
        else:
            st.success(f"Likely to be KEPT — cancellation probability {prob:.1%}")

with tab2:
    if st.button("Predict ADR"):
        X = prepare(booking, reg_columns, scaler_reg)
        rate = lin_model.predict(X)[0]
        st.info(f"Predicted average daily rate: **{max(rate, 0):.2f}**")
