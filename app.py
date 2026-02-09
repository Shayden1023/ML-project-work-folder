import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests

# ----------------------------
# Load pre-trained model + columns
# ----------------------------
def load_model():
    file_id = "1M22pQzZvEmqJ9tp0EJSQ6vKgN10BOsIH"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "ford.pkl"

    if not os.path.exists(output):
        with st.spinner("Downloading model from Google Drive..."):
            response = requests.get(url)
            with open(output, "wb") as f:
                f.write(response.content)

    return joblib.load(output)

model = load_model()
train_columns = joblib.load("ford_columns.pkl")

# ----------------------------
# Feature engineering function
# ----------------------------
def feature_engineer(df: pd.DataFrame, current_year: int = 2026) -> pd.DataFrame:
    df = df.copy().drop_duplicates()

    for col in ["model", "transmission", "fuelType"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["car_age"] = current_year - df["year"]
    df["car_age"] = df["car_age"].clip(lower=0)

    df["mileage_per_year"] = df["mileage"] / (df["car_age"] + 1)
    df["mpg_per_engine"] = df["mpg"] / (df["engineSize"] + 0.1)

    df["tax_band"] = pd.cut(
        df["tax"],
        bins=[-np.inf, 0, 100, 200, 500, np.inf],
        labels=["Zero", "Low", "Medium", "High", "Very High"],
        include_lowest=True
    ).astype("category")

    df.drop("year", axis=1, inplace=True)
    return df

# ----------------------------
# App layout
# ----------------------------
st.set_page_config(page_title="Ford Price Predictor", layout="wide")
st.title("Ford Car Price Predictor (Pre-trained Random Forest)")

st.write("This app uses a pre-trained Random Forest model (`ford.pkl`) to predict car prices based on user input.")

# ----------------------------
# Prediction form
# ----------------------------
st.subheader("Predict a Car Price")

models = [' Fiesta', ' Focus', ' Puma', ' Kuga', ' EcoSport', ' C-MAX',
       ' Mondeo', ' Ka+', ' Tourneo Custom', ' S-MAX', ' B-MAX', ' Edge',
       ' Tourneo Connect', ' Grand C-MAX', ' KA', ' Galaxy', ' Mustang',
       ' Grand Tourneo Connect', ' Fusion', ' Ranger', ' Streetka',
       ' Escort', ' Transit Tourneo', 'Focus']
transmissions = ["Manual", "Automatic", "Semi-Auto"]
fuel_types = ["Petrol", "Diesel", "Hybrid", "Electric"]

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        model_in = st.selectbox("Model", models)
        transmission_in = st.selectbox("Transmission", transmissions)
        fuel_in = st.selectbox("Fuel Type", fuel_types)

    with c2:
        year_in = st.number_input("Year", min_value=1990, max_value=2026, value=2018, step=1)
        mileage_in = st.slider("Mileage", min_value=0, max_value=100000, value=20000, step=1000)

    with c3:
        tax_in = st.slider("Tax", min_value=0, max_value=500, value=150, step=1)
        mpg_in = st.slider("MPG", min_value=0.0, max_value=100.0, value=55.0, step=0.1)
        engine_in = st.slider("Engine Size", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    # Custom button styling
    st.markdown(
        """
        <style>
        div.stFormSubmitButton > button {
            background-color: #2E86C1;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
        }
        div.stFormSubmitButton > button:hover {
            background-color: #1B4F72;
            color: #f9f9f9;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    submitted = st.form_submit_button("Predict Price")
new_fe=pd.DataFrame()
if submitted:
    new_data = pd.DataFrame({
        "model": [str(model_in).strip()],
        "year": [int(year_in)],
        "transmission": [str(transmission_in).strip()],
        "mileage": [int(mileage_in)],
        "fuelType": [str(fuel_in).strip()],
        "tax": [int(tax_in)],
        "mpg": [float(mpg_in)],
        "engineSize": [float(engine_in)]
    })

    new_fe = feature_engineer(new_data)

    # Only encode categorical columns if they exist
    cat_cols = new_fe.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        new_encoded = pd.get_dummies(new_fe, columns=cat_cols, drop_first=True)
    else:
        new_encoded = new_fe.copy()

    # Align to training columns
    new_encoded = new_encoded.reindex(columns=train_columns, fill_value=0)

    pred_price = model.predict(new_encoded)[0]

    # Styled prediction card
    st.markdown(
        f"""
        <div style="background-color:#2E86C1;
                    padding:20px;
                    border-radius:10px;
                    text-align:center;
                    color:white;
                    font-size:26px;
                    font-weight:bold;
                    margin-top:20px;">
            Predicted Price: £{pred_price:,.0f}
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# Feature importance report
# ----------------------------
importance = pd.Series(model.feature_importances_, index=train_columns).sort_values(ascending=False)

with st.expander("Top 5 Most Important Features Report"):
    top5 = importance.head(5).reset_index()
    top5.columns = ["Feature", "Importance"]

    # Display as a table
    st.table(top5)

    # Narrative report
    st.markdown("### Report Summary")
    for i, row in top5.iterrows():
        st.markdown(
            f"- **{row['Feature']}** contributes significantly to price prediction "
            f"with an importance score of **{row['Importance']:.4f}**."
        )

    # Highlight the most important feature
    most_important = top5.iloc[0]
    st.markdown(
        f"The most influential feature is **{most_important['Feature']}**, "
        f"indicating it plays the largest role in determining car prices."
    )