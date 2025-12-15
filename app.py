import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from io import StringIO

# Panjang sinyal yang digunakan di training (berdasarkan kolom att1 hingga att18530)
EXPECTED_LENGTH = 18530 
TARGET_CLASSES = {0: "Abnormal", 1: "Normal"}

st.title("Abnormal Heartbeat Classification")

# --- 1. Memuat Model dan Scaler ---
@st.cache_resource
def load_resources():
    try:
        model = load_model("cnn_abnormal_heartbeat.h5")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Pastikan file 'cnn_abnormal_heartbeat.h5' dan 'scaler.pkl' ada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat sumber daya: {e}")
        st.stop()

model, scaler = load_resources()

# --- 2. Input Data ---
st.header("Pilih Metode Input Data")
input_method = st.radio(
    "Pilih cara Anda ingin memasukkan data:",
    ('Unggah File CSV', 'Input Data Manual (Paste CSV)'),
    index=0 
)

df = None

if input_method == 'Unggah File CSV':
    uploaded = st.file_uploader("Unggah File CSV (1 sampel heartbeat)", type="csv")
    if uploaded:
        # Asumsi CSV tanpa header (hanya nilai sinyal)
        df = pd.read_csv(uploaded, header=None) 
        st.success("File berhasil diunggah.")

elif input_method == 'Input Data Manual (Paste CSV)':
    st.info("Tempelkan data numerik sinyal heartbeat Anda di bawah ini, dipisahkan oleh koma, baris baru, atau spasi.")

    text_input = st.text_area(
        "Input Data CSV Mentah (misal: 0.1, 0.2, 0.3, ...)",
        height=200
    )

    if text_input:
        try:
            # Fleksibel membaca pemisah (koma, spasi, atau baris baru)
            df = pd.read_csv(StringIO(text_input), sep=r'\s*,\s*|\s+', engine='python', header=None)
            st.success(f"Data berhasil diproses. Ditemukan {df.size} nilai.")
        except Exception as e:
            st.error(f"Error dalam memproses data: {e}")

# --- 3. Proses Prediksi ---
if df is not None:

    # 3a. Data Preparation & Flattening
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    elif df.shape[0] > 1:
        # Flatten data menjadi satu baris (sample)
        st.warning("Data diinterpretasikan sebagai multiple baris. Data akan di-flatten menjadi satu sampel.")
        df = pd.DataFrame(df.values.flatten()).T

    # 3b. Validasi Panjang
    if df.shape[1] != EXPECTED_LENGTH:
        st.error(f"Error: Panjang sinyal ({df.shape[1]}) tidak sesuai dengan panjang yang diharapkan ({EXPECTED_LENGTH}).")
        st.stop()

    st.subheader("Hasil Analisis")
    st.write(f"Total Panjang Sinyal: {df.shape[1]}")

    try:
        # 3c. Scaling Data
        scaled = scaler.transform(df.values) 

        # 3d. Reshaping untuk CNN (dari (1, L) menjadi (1, L, 1))
        reshaped = np.expand_dims(scaled, axis=-1)

        # 3e. Prediksi
        with st.spinner('Melakukan Klasifikasi...'):
            pred = model.predict(reshaped)
            label = np.argmax(pred)

            # Indeks 0 = Abnormal, Indeks 1 = Normal
            prob_abnormal = pred[0][0] * 100 
            prob_normal = pred[0][1] * 100   

        st.markdown("---")

        # 3f. Menampilkan Hasil dengan Logika yang BENAR (0=Abnormal)
        predicted_class = TARGET_CLASSES[label]

        if predicted_class == "Normal":
            st.success(f"## 🟢 Prediksi: {predicted_class} (Keyakinan: {prob_normal:.2f}%)")
        else:
            st.error(f"## 🔴 Prediksi: {predicted_class} (Keyakinan: {prob_abnormal:.2f}%)")

        # Menampilkan probabilitas detail
        st.markdown(f"**Probabilitas:**")
        st.write(f"- Normal: {prob_normal:.2f}%")
        st.write(f"- Abnormal: {prob_abnormal:.2f}%")

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
        st.warning("Pastikan data sudah benar dan scaler.pkl cocok dengan data input.")
