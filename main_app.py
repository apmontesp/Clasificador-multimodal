import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from streamlit_drawable_canvas import st_canvas
import cv2

st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")

st.title("🔢 Plataforma de Clasificación MNIST")
st.markdown("Compara modelos de ML y prueba tu propia escritura a mano.")

# --- 1. CARGA Y CALIDAD DE DATOS ---
digits = load_digits()
X, y = digits.data, digits.target

with st.sidebar:
    st.header("Configuración")
    test_size = st.slider("Porcentaje de Test", 10, 50, 20) / 100
    use_pca = st.checkbox("¿Usar PCA? (Reducción de dimensionalidad)")
    cv_strategy = st.selectbox("Estrategia de Validación Cruzada", ["StratifiedKFold", "KFold"])
    n_splits = st.slider("Folds (K)", 2, 10, 5)

# --- 2. PROCESAMIENTO ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = None
if use_pca:
    pca = PCA(n_components=0.95)
    X_proc = pca.fit_transform(X_scaled)
    st.sidebar.info(f"Dimensiones reducidas de 64 a {X_proc.shape[1]}")
else:
    X_proc = X_scaled

X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_size, random_state=42)

# --- 3. MODELADO ---
models = {
    "ANN (Perceptrón Multicapa)": MLPClassifier(max_iter=500),
    "KNN (Vecinos Cercanos)": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM (Soporte Vectorial)": SVC(probability=True),
    "Random Forest": RandomForestClassifier()
}

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Calidad y Desempeño")
    selected_model_name = st.selectbox("Selecciona un modelo para evaluar", list(models.keys()))
    model = models[selected_model_name]
    
    # Validación Cruzada
    cv = StratifiedKFold(n_splits=n_splits) if cv_strategy == "StratifiedKFold" else KFold(n_splits=n_splits)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    
    # Entrenamiento y Predicción
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    # Métricas
    st.write(f"**Exactitud en Train:** {train_acc:.4f}")
    st.write(f"**Exactitud en Test:** {test_acc:.4f}")
    st.write(f"**CV Mean Accuracy ({n_splits} folds):** {cv_scores.mean():.4f}")

    # Matriz de Confusión
    fig_cm, ax_cm = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title(f"Matriz de Confusión: {selected_model_name}")
    st.pyplot(fig_cm)

# --- 4. DIBUJO CON EL MOUSE ---
with col2:
    st.subheader("✍️ ¡Dibuja un número!")
    st.write("Dibuja un dígito en el recuadro y presiona 'Predecir'.")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", 
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Reconocer Dígito"):
        if canvas_result.image_data is not None:
            # Procesar imagen del canvas a formato 8x8 de sklearn
            img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
            img_resizing = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
            
            # Normalizar a rango 0-16 (como MNIST de sklearn)
            img_normalized = (img_resizing / 255.0) * 16
            
            # Transformaciones de escalado y PCA si aplica
            input_data = img_normalized.flatten().reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            
            if use_pca:
                input_scaled = pca.transform(input_scaled)
            
            prediction = model.predict(input_scaled)
            probs = model.predict_proba(input_scaled) if hasattr(model, "predict_proba") else None
            
            st.success(f"### 🤖 Predicción: {prediction[0]}")
            
            if probs is not None:
                prob_df = pd.DataFrame(probs, columns=range(10))
                st.bar_chart(prob_df.T)
