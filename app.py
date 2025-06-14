import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Muat Model dan Scaler
with open("model_obesitas.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Daftar fitur sesuai saat training
features = [
    'Gender', 'Age', 'Height', 'Weight',
    'family_history_with_overweight', 'FAVC',
    'FCVC', 'NCP', 'CAEC', 'SMOKE',
    'CH2O', 'SCC', 'FAF', 'TUE',
    'CALC', 'MTRANS'
]

# Mapping input
binary_mapping = {"Ya": 1, "Tidak": 0}
gender_mapping = {"Laki-laki": 1, "Perempuan": 0}
caec_mapping = {"Tidak Pernah": 0, "Kadang-kadang": 1, "Sering": 2, "Selalu": 3}
calc_mapping = {"Tidak Pernah": 0, "Kadang-kadang": 1, "Sering": 2, "Selalu": 3}
mtrans_mapping = {"Mobil Pribadi": 0, "Motor": 1, "Sepeda": 2, "Transportasi Umum": 3, "Jalan Kaki": 4}
label_mapping = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}

st.set_page_config(page_title='Prediksi Obesitas', layout='centered')
st.title("🎯 Prediksi Obesitas Berdasarkan Gaya Hidup")
st.write("Silakan isi data gaya hidupmu di bawah ini untuk prediksi kategori obesitas:")


def user_input():
    input_data = {}
    for feature in features:
        if feature == 'Gender':
            val = st.selectbox("Jenis Kelamin", list(gender_mapping.keys()))
            input_data[feature] = gender_mapping[val]
        elif feature == 'Age':
            input_data[feature] = st.slider("Usia (tahun)", 10, 100, 25)
        elif feature == 'Height':
            input_data[feature] = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, value=1.7)
        elif feature == 'Weight':
            input_data[feature] = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0)
        elif feature == 'FCVC':
            input_data[feature] = st.slider("Frekuensi konsumsi sayur per hari", 1.0, 3.0, step=0.1)
        elif feature == 'NCP':
            input_data[feature] = st.slider("Jumlah makan per hari", 1.0, 4.0, step=0.1)
        elif feature == 'CH2O':
            input_data[feature] = st.slider("Jumlah air yang diminum per hari (liter)", 1.0, 3.0, step=0.1)
        elif feature == 'FAF':
            input_data[feature] = st.slider("Jumlah olahraga per minggu", 0.0, 3.0, step=0.1)
        elif feature == 'TUE':
            input_data[feature] = st.slider("Jumlah waktu di layar per hari (jam)", 0.0, 2.0, step=0.1)
        elif feature == 'FAVC':
            val = st.selectbox("Makan makanan kalori Tinggi?", list(binary_mapping.keys()))
            input_data[feature] = binary_mapping[val]
        elif feature == 'SMOKE':
            val = st.selectbox("Merokok?", list(binary_mapping.keys()))
            input_data[feature] = binary_mapping[val]
        elif feature == 'SCC':
            val = st.selectbox("Memantau kalori yang dimakan?", list(binary_mapping.keys()))
            input_data[feature] = binary_mapping[val]
        elif feature == 'CAEC':
            val = st.selectbox("Ngemil?", list(caec_mapping.keys()))
            input_data[feature] = caec_mapping[val]
        elif feature == 'CALC':
            val = st.selectbox("Minum alkohol?", list(calc_mapping.keys()))
            input_data[feature] = calc_mapping[val]
        elif feature == 'MTRANS':
            val = st.selectbox("Alat transportasi yang digunakan?", list(mtrans_mapping.keys()))
            input_data[feature] = mtrans_mapping[val]

    return pd.DataFrame([input_data])

# Input & Prediksi
input_df = user_input()

if st.button("Prediksi"):
    try:
        # Mengikuti urutan yang sesuai
        input_df = input_df[features]

        # Transform dan prediksi
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)
        pred_label = label_mapping.get(pred[0], "Tidak diketahui")

        st.success(f"Kategori Obesitas Anda: {pred_label}")
        st.markdown("#### Data Anda:")
        st.write(input_df)
    except Exception as e:
        st.error("❌ Terjadi error saat memproses input.")
        st.exception(e)
