import streamlit as st
import numpy as np
import pandas as pd
import cv2
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="DIP & Digit Classifier")

# --- FUNCIONES DE DIP (Digital Image Processing) ---
def process_user_image(canvas_data, blur_k, stroke_w):
    """
    Pipeline de DIP: 
    1. Conversión a Grises -> 2. Recorte de bordes (Bbox) -> 
    3. Suavizado -> 4. Normalización a escala 0-16.
    """
    # 1. Convertir RGBA a Gris
    img = cv2.cvtColor(canvas_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    
    # 2. Encontrar el Bounding Box del dibujo (para centrar)
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Extraer solo la zona dibujada (evita información vacía)
        img = img[y:y+h, x:x+w]
        # Añadir un pequeño padding para que no toque los bordes
        img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    
    # 3. Suavizado (Gaussian Blur) para evitar bordes duros (Aliasing)
    # Esto ayuda a que el modelo no se confunda con píxeles aislados
    if blur_k % 2 == 0: blur_k += 1 # Kernel debe ser impar
    img_smooth = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    
    # 4. Redimensionar a 8x8 (Formato sklearn)
    img_8x8 = cv2.resize(img_smooth, (8, 8), interpolation=cv2.INTER_AREA)
    
    # 5. Escalar de 0-255 a 0-16 (Rango del dataset original)
    img_final = (img_8x8 / 255.0) * 16
    
    return img_final

# --- CONFIGURACIÓN DE DATOS ---
digits = load_digits()
X, y = digits.data, digits.target

st.title("🔢 Clasificador MNIST con Procesamiento Digital de Imágenes (DIP)")

with st.sidebar:
    st.header("1. Parámetros de Datos")
    test_pct = st.slider("Porcentaje de Test", 10, 50, 20) / 100
    use_pca = st.toggle("Aplicar PCA (95% Varianza)")
    
    st.header("2. Parámetros de Dibujo (DIP)")
    stroke_width = st.slider("Grosor del Lápiz", 5, 30, 15)
    blur_kernel = st.slider("Suavizado (Blur)", 1, 15, 7, step=2)

# Escalado previo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA opcional
pca = None
if use_pca:
    pca = PCA(n_components=0.95)
    X_proc = pca.fit_transform(X_scaled)
else:
    X_proc = X_scaled

X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_pct, random_state=42)

# --- MODELADO ---
model_choice = st.selectbox("Seleccione Modelo", ["ANN", "KNN", "Bayes", "SVM", "Random Forest"])
models = {
    "ANN": MLPClassifier(max_iter=500, hidden_layer_sizes=(64, 32)),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Bayes": GaussianNB(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier()
}

clf = models[model_choice]
clf.fit(X_train, y_train)

# --- INTERFAZ PRINCIPAL ---
col_ui, col_viz = st.columns([1, 1])

with col_ui:
    st.subheader("Dibujo y Reconocimiento")
    canvas = st_canvas(
        fill_color="black",
        stroke_width=stroke_width,
        stroke_color="white",
        background_color="black",
        height=250,
        width=250,
        drawing_mode="freedraw",
        key="mnist_canvas"
    )

    if st.button("Procesar y Clasificar"):
        if canvas.image_data is not None:
            # Aplicar Pipeline DIP
            processed_img = process_user_image(canvas.image_data, blur_kernel, stroke_width)
            
            # Mostrar miniatura procesada
            st.write("Vista 8x8 (DIP Output):")
            fig_mini, ax_mini = plt.subplots(figsize=(2,2))
            ax_mini.imshow(processed_img, cmap='gray')
            ax_mini.axis('off')
            st.pyplot(fig_mini)
            
            # Predicción
            flat_img = processed_img.flatten().reshape(1, -1)
            scaled_img = scaler.transform(flat_img)
            if use_pca: scaled_img = pca.transform(scaled_img)
            
            pred = clf.predict(scaled_img)
            st.success(f"### Predicción Final: {pred[0]}")

with col_viz:
    st.subheader("Desempeño del Modelo")
    acc = clf.score(X_test, y_test)
    st.metric("Exactitud en Test", f"{acc:.2%}")
    
    # Matriz de Confusión
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', ax=ax_cm)
    st.pyplot(fig_cm)
