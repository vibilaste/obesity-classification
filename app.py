
import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Prediksi Tingkat Obesitas")

# Input
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.number_input("Usia", min_value=10, max_value=100, value=25)
height = st.number_input("Tinggi Badan (m)", min_value=1.0, max_value=2.5, value=1.70)
weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0)
family_history = st.selectbox("Riwayat keluarga kelebihan berat badan", ["yes", "no"])
favc = st.selectbox("Sering makan tinggi kalori?", ["yes", "no"])
fcvc = st.slider("Frekuensi makan sayur (0–3)", 0.0, 3.0, 2.0)
ncp = st.slider("Jumlah makan besar/hari", 1.0, 5.0, 3.0)
caec = st.selectbox("Camilan", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Merokok", ["yes", "no"])
ch2o = st.slider("Air per hari (liter)", 0.0, 3.0, 2.0)
scc = st.selectbox("Kontrol kalori?", ["yes", "no"])
faf = st.slider("Aktivitas fisik (jam/minggu)", 0.0, 3.0, 1.0)
tue = st.slider("Waktu layar (jam/hari)", 0.0, 3.0, 2.0)
calc = st.selectbox("Alkohol", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportasi", ["Public_Transportation", "Walking", "Automobile", "Bike", "Motorbike"])

if st.button("Prediksi"):
    mapping = {
        "yes": 1, "no": 0,
        "Male": 1, "Female": 0,
        "no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3,
        "Public_Transportation": 0, "Walking": 1, "Automobile": 2, "Bike": 3, "Motorbike": 4
    }

    data = np.array([[
        mapping[gender],
        age,
        height,
        weight,
        mapping[family_history],
        mapping[favc],
        fcvc,
        ncp,
        mapping[caec],
        mapping[smoke],
        ch2o,
        mapping[scc],
        faf,
        tue,
        mapping[calc],
        mapping[mtrans]
    ]])

    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    st.success(f"Tingkat Obesitas Anda: **{pred}**")
