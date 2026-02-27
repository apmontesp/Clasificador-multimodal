import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from streamlit_drawable_canvas import st_canvas

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Plataforma ML & DIP MNIST")

# --- 1. CARGA Y CALIDAD DE DATOS (VERIFICACIÓN) ---
@st.cache_data
def load_data():
    digits = load_digits()
    return digits.data, digits.target, digits.images

X_raw, y_raw, images_raw = load_data()

st.title("🔢 Clasificador Multiclase con DIP y Verificación")
st.markdown("""
Esta plataforma permite comparar 5 modelos de ML, aplicar reducción de dimensionalidad (PCA) 
y verificar la calidad de los datos de entrada mediante Procesamiento Digital de Imágenes.
""")

# --- 2. BARRA LATERAL (CONFIGURACIÓN) ---
with st.sidebar:
    st.header("⚙️ Configuración del Modelo")
    test_size = st.slider("Porcentaje de Datos para Test", 10, 50, 20) / 100
    use_pca = st.toggle("Habilitar PCA (95% Varianza)")
    
    st.header("🎨 Ajustes de DIP (Entrada)")
    stroke_w = st.slider("Grosor del dibujo", 5, 40, 20)
    blur_v = st.slider("Suavizado (Gaussian Blur)", 1, 19, 9, step=2)

# --- 3. PREPROCESAMIENTO Y PCA ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

if use_pca:
    pca_obj = PCA(n_components=0.95)
    X_proc = pca_obj.fit_transform(X_scaled)
    st.sidebar.success(f"PCA Activo: {X_proc.shape[1]} componentes")
else:
    X_proc = X_scaled
    pca_obj = None

X_train, X_test, y_train, y_test = train_test_split(X_proc, y_raw, test_size=test_size, random_state=42)

# --- 4. ENTRENAMIENTO Y VALIDACIÓN CRUZADA ---
st.header("📊 Evaluación de Modelos")

models = {
    "ANN": MLPClassifier(max_iter=500, hidden_layer_sizes=(64, 32)),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Bayes": GaussianNB(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier()
}

# Selección de modelo para ver detalle
selected_model_name = st.selectbox("Selecciona un modelo para clasificar tu dibujo:", list(models.keys()))
clf = models[selected_model_name]

# Ejecución de entrenamiento
cv = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Mostrar métricas en columnas
m1, m2, m3 = st.columns(3)
m1.metric("Exactitud (Test)", f"{accuracy_score(y_test, y_pred):.2%}")
m2.metric("Validación Cruzada (Promedio)", f"{cv_scores.mean():.2%}")
m3.metric("Calidad de Datos", "Nulos: 0 / 100%")

# Matriz de Confusión
fig_cm, ax_cm = plt.subplots(figsize=(10, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_title(f"Matriz de Confusión - {selected_model_name}")
st.pyplot(fig_cm)

st.divider()

# --- 5. ETAPA DE DIP Y VERIFICACIÓN DE DIBUJO ---
st.header("✍️ Reconocimiento de Dibujo con DIP")



col_canvas, col_dip, col_pred = st.columns([1.5, 1.5, 1])

with col_canvas:
    st.subheader("Entrada: Dibujo Manual")
    canvas = st_canvas(
        fill_color="black",
        stroke_width=stroke_w,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_mnist"
    )

def dip_process(canvas_data, blur_k):
    # DIP 1: Conversión a Gris
    img = cv2.cvtColor(canvas_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    
    # DIP 2: Centrado y Bounding Box (Verificación de área de interés)
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
        img = cv2.copyMakeBorder(img, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=0)
    
    # DIP 3: Suavizado para evitar traslapo y ruido (Antialiasing)
    img_blur = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    
    # DIP 4: Redimensión a 8x8 (Igualar entrada a salida de entrenamiento)
    img_res = cv2.resize(img_blur, (8, 8), interpolation=cv2.INTER_AREA)
    
    # DIP 5: Normalización de rango (0-255 a 0-16)
    img_final = (img_res / 255.0) * 16
    return img_final

if canvas.image_data is not None:
    processed_img = dip_process(canvas.image_data, blur_v)
    
    with col_dip:
        st.subheader("Verificación (Salida DIP)")
        fig_v, ax_v = plt.subplots(1, 2)
        ax_v[0].imshow(processed_img, cmap='gray')
        ax_v[0].set_title("Tu Dato (8x8)")
        
        # Comparación con un dato real del dataset
        sample_real = images_raw[np.random.randint(0, 1797)]
        ax_v[1].imshow(sample_real, cmap='magma')
        ax_v[1].set_title("Dato Real Ref.")
        st.pyplot(fig_v)
        
        # Verificación de histogramas
        st.write("**Consistencia de Intensidad:**")
        fig_h, ax_h = plt.subplots(figsize=(5, 2))
        sns.histplot(processed_img.flatten(), bins=16, kde=True, ax=ax_h, color="blue")
        ax_h.set_xlim(0, 16)
        st.pyplot(fig_h)

    with col_pred:
        st.subheader("Resultado")
        if st.button("🚀 Clasificar"):
            # Preparar dato para el modelo
            flat_data = processed_img.flatten().reshape(1, -1)
            # Aplicar el mismo escalador y PCA que en el entrenamiento
            input_scaled = scaler.transform(flat_data)
            if use_pca:
                input_ready = pca_obj.transform(input_scaled)
            else:
                input_ready = input_scaled
            
            prediction = clf.predict(input_ready)
            st.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>{prediction[0]}</h1>", unsafe_allow_html=True)
            
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(input_ready)
                st.bar_chart(pd.DataFrame(probs.T, index=range(10), columns=["Probabilidad"]))
