import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Judul aplikasi
st.title("Prediksi Tingkat Obesitas")
st.write("Masukkan data untuk memprediksi tingkat obesitas menggunakan model pilihan Anda.")

# Pilih model
st.header("Pilih Model")
model_choice = st.selectbox("Pilih Model untuk Prediksi", ["XGBoost", "Random Forest", "Logistic Regression"])
if model_choice == "XGBoost":
    model = joblib.load('model.pkl')
elif model_choice == "Random Forest":
    model = joblib.load('rf_model.pkl')
else:
    model = joblib.load('lr_model.pkl')

# Memuat scaler
scaler = joblib.load('scaler.pkl')

# Definisikan fitur yang digunakan
fitur = ['Weight', 'Height', 'Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
         'Gender', 'family_history_with_overweight', 'CAEC_Frequently',
         'CAEC_Sometimes', 'MTRANS_Public_Transportation', 'MTRANS_Walking']

# Definisikan mapping untuk kelas target
class_mapping = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II',
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
}

# Input untuk fitur numerik
st.header("Input Fitur Numerik")
age = st.number_input("Usia (tahun, 0-120)", min_value=0.0, max_value=120.0, value=25.0)
height = st.number_input("Tinggi (meter, 1.0-2.5)", min_value=1.0, max_value=2.5, value=1.75)
weight = st.number_input("Berat (kg, 20-300)", min_value=20.0, max_value=300.0, value=70.0)
fcvc = st.number_input("Frekuensi konsumsi sayur (1-3)", min_value=1.0, max_value=3.0, value=2.0)
ncp = st.number_input("Jumlah makan per hari (1-4)", min_value=1.0, max_value=4.0, value=3.0)
ch2o = st.number_input("Konsumsi air (liter, 1-3)", min_value=1.0, max_value=3.0, value=2.0)
faf = st.number_input("Frekuensi aktivitas fisik (0-3)", min_value=0.0, max_value=3.0, value=1.0)
tue = st.number_input("Waktu penggunaan teknologi (jam, 0-2)", min_value=0.0, max_value=2.0, value=1.0)

# Input untuk fitur kategorikal
st.header("Input Fitur Kategorikal")
gender = st.selectbox("Jenis Kelamin", options=['Male', 'Female'])
family_history = st.selectbox("Riwayat keluarga dengan obesitas", options=['Yes', 'No'])
caec = st.selectbox("Frekuensi makan camilan", options=['Sometimes', 'Frequently', 'Always', 'no'])
mtrans = st.selectbox("Moda transportasi utama", options=['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

# Tombol untuk prediksi
if st.button("Prediksi"):
    # Encoding input kategorikal
    gender_encoded = 1 if gender == 'Male' else 0
    family_history_encoded = 1 if family_history == 'Yes' else 0
    caec_frequently = 1 if caec == 'Frequently' else 0
    caec_sometimes = 1 if caec == 'Sometimes' else 0
    mtrans_public = 1 if mtrans == 'Public_Transportation' else 0
    mtrans_walking = 1 if mtrans == 'Walking' else 0

    # Buat array input
    input_data = np.array([[weight, height, age, fcvc, ncp, ch2o, faf, tue,
                            gender_encoded, family_history_encoded,
                            caec_frequently, caec_sometimes,
                            mtrans_public, mtrans_walking]])

    # Standarisasi fitur numerik
    input_data[:, :8] = scaler.transform(input_data[:, :8])

    # Prediksi
    prediction = model.predict(input_data)
    predicted_class = class_mapping[prediction[0]]

    # Tampilkan hasil
    st.success(f"Prediksi tingkat obesitas menggunakan {model_choice}: **{predicted_class}**")

# Visualisasi distribusi kelas
st.header("Distribusi Kelas Dataset")
df = pd.read_csv('ObesityDataSet_Preprocessed.csv')
st.bar_chart(df['NObeyesdad'].value_counts().rename(index=class_mapping))

# Visualisasi feature importance (hanya untuk Random Forest dan XGBoost)
if model_choice in ["Random Forest", "XGBoost"]:
    st.header("Pentingnya Fitur dalam Prediksi")
    feature_importance = pd.Series(model.feature_importances_, index=fitur).sort_values(ascending=False)
    st.bar_chart(feature_importance)

# Panduan pengguna
st.markdown("""
### Panduan Penggunaan
1. Pilih model yang ingin digunakan (XGBoost, Random Forest, atau Logistic Regression).
2. Masukkan nilai untuk fitur numerik (misalnya, usia, tinggi, berat) dalam rentang yang ditentukan.
3. Pilih opsi yang sesuai untuk fitur kategorikal (misalnya, jenis kelamin, riwayat keluarga).
4. Klik tombol "Prediksi" untuk melihat hasil klasifikasi tingkat obesitas.
5. Lihat distribusi kelas dataset dan pentingnya fitur (untuk Random Forest dan XGBoost) di bagian bawah.
""")

# CSS kustom untuk tampilan
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; padding: 10px 20px; }
    .stNumberInput input { background-color: #ffffff; border: 1px solid #4CAF50; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)
