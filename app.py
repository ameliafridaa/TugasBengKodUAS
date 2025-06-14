import pickle
import pandas as pd

# ---------------------------
# 1. Load model & scaler
# ---------------------------
try:
    with open("/mount/src/tugasbengkoduas/model_obesitas.pkl", "rb") as f:
        model = pickle.load(f)
    with open("/mount/src/tugasbengkoduas/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("❌ File model atau scaler tidak ditemukan.")
    exit()
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    exit()

# ---------------------------
# 2. Data input pengguna (contoh)
# ---------------------------
# Misal input_df dibentuk dari input user, atau form, atau file
input_df = pd.DataFrame([{
    "feature1": 5.2,
    "feature2": 3.1,
    "feature3": 0.8,
    # ... tambahkan semua fitur sesuai model
}])

# ---------------------------
# 3. Cocokkan urutan kolom dengan scaler
# ---------------------------
try:
    input_df = input_df[scaler.feature_names_in_]  # urutkan dan pastikan kolom cocok
    input_scaled = scaler.transform(input_df)
except KeyError as e:
    print(f"❌ Kolom input tidak sesuai: {e}")
    print(f"Diperlukan kolom: {scaler.feature_names_in_}")
    exit()
except Exception as e:
    print(f"❌ Gagal mentransformasi data: {e}")
    exit()

# ---------------------------
# 4. Prediksi
# ---------------------------
try:
    prediction = model.predict(input_scaled)
    print("✅ Hasil prediksi:", prediction)
except Exception as e:
    print(f"❌ Gagal melakukan prediksi: {e}")
