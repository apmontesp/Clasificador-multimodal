import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from streamlit_drawable_canvas import st_canvas

# Configuración inicial
st.set_page_config(layout="wide", page_title="DIP Verification System")

# --- CARGA DE DATOS REFERENCIA ---
digits = load_digits()
X_ref, y_ref = digits.data, digits.target

# --- PIPELINE DE PROCESAMIENTO (DIP) ---
def dip_pipeline(canvas_data, blur_k):
    # 1. Grises
    img = cv2.cvtColor(canvas_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    # 2. Bounding Box para centrar (Verificación de forma)
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
        img = cv2.copyMakeBorder(img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=0)
    # 3. Suavizado (Antialiasing)
    img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    # 4. Redimensión y Normalización (0-16)
    img_8x8 = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    img_final = (img_8x8 / 255.0) * 16
    return img_final

# --- INTERFAZ ---
st.title("🔍 Verificación de Calidad y Consistencia de Datos")
st.info("El objetivo es que el 'Dato Procesado' tenga una distribución de varianza similar al 'Dato Real' de SKLearn.")

with st.sidebar:
    st.header("Control de Entrenamiento")
    test_size = st.slider("Test Size %", 10, 50, 20) / 100
    model_type = st.selectbox("Modelo", ["ANN", "KNN", "SVM", "Bayes", "Random Forest"])
    
    st.header("Ajuste de DIP")
    stroke_w = st.slider("Grosor de trazo", 5, 40, 20)
    blur_v = st.slider("Suavizado (Blur)", 1, 19, 9, step=2)

# Entrenamiento rápido
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ref)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_ref, test_size=test_size)

models = {"ANN": MLPClassifier(), "KNN": KNeighborsClassifier(), "SVM": SVC(), "Bayes": GaussianNB(), "Random Forest": RandomForestClassifier()}
clf = models[model_type].fit(X_train, y_train)

# --- ÁREA DE PRUEBAS ---
col_input, col_verify = st.columns([1, 1])

with col_input:
    st.subheader("Entrada: Dibujo Manual
