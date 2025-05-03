import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# --- Configuración general ---
st.set_page_config(page_title="🌼 Detector de Flores", layout="wide")

# --- Cargar modelo y clases ---
modelo = load_model("modelo_flores.h5")
CLASES = ['Margarita', 'Girasol', 'Rosa', 'Tulipan']

# --- Información botánica ---
INFO_FLORES = {
    'Margarita': {
        'nombre_cientifico': 'Bellis perennis',
        'origen': 'Europa y Asia templada',
        'uso': 'Ornamental y medicinal',
        'curiosidad': 'Simboliza inocencia y pureza.',
        'imagen_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Bellis_perennis_white.jpg/800px-Bellis_perennis_white.jpg'
    },
    'Girasol': {
        'nombre_cientifico': 'Helianthus annuus',
        'origen': 'América del Norte',
        'uso': 'Aceite, ornamental, comestible',
        'curiosidad': 'Sigue al sol durante el día.',
        'imagen_url': 'https://upload.wikimedia.org/wikipedia/commons/4/40/Sunflower_sky_backdrop.jpg'
    },
    'Rosa': {
        'nombre_cientifico': 'Rosa spp.',
        'origen': 'Asia principalmente',
        'uso': 'Ornamental, perfumería, medicina',
        'curiosidad': 'Existen más de 100 especies.',
        'imagen_url': 'https://upload.wikimedia.org/wikipedia/commons/b/bf/Red_rose.jpg'
    },
    'Tulipan': {
        'nombre_cientifico': 'Tulipa',
        'origen': 'Turquía y Asia Central',
        'uso': 'Ornamental',
        'curiosidad': 'Símbolo de riqueza en los Países Bajos.',
        'imagen_url': 'https://upload.wikimedia.org/wikipedia/commons/4/45/Red_tulip.jpg'
    }
}

# --- Estilo CSS personalizado oscuro ---
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
        }
        .titulo {
            font-size: 40px;
            font-weight: bold;
            color: #ff4b4b;
        }
        .info-box {
            background-color: #1c1c1c;
            border-left: 5px solid #ff4b4b;
            padding: 20px;
            border-radius: 10px;
            color: #eeeeee;
        }
        h2 {
            color: #ff4b4b;
        }
    </style>
""", unsafe_allow_html=True)

# --- Título ---
st.markdown("<div class='titulo'>🌸 Detector Inteligente de Flores</div>", unsafe_allow_html=True)
st.write("Sube una imagen o usa tu cámara para identificar la flor y obtener datos útiles.")

# --- Selector de entrada ---
origen = st.radio("Elige cómo cargar la imagen:", ["📁 Subir archivo", "📷 Usar cámara"], horizontal=True)

imagen = None
if origen == "📁 Subir archivo":
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if archivo:
        imagen = Image.open(archivo).convert("RGB")
elif origen == "📷 Usar cámara":
    camara = st.camera_input("Toma una foto")
    if camara:
        imagen = Image.open(camara).convert("RGB")

# --- Clasificación e interfaz en dos columnas ---
if imagen:
    img_resized = imagen.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = modelo.predict(img_array)[0]
    prob = np.max(pred)
    idx = np.argmax(pred)
    clase = CLASES[idx]

    col_info, col_img = st.columns([1.2, 1])

    with col_info:
        if prob >= 0.80:
            st.markdown(f"<h2>✅ Flor detectada: {clase} ({prob:.1%})</h2>", unsafe_allow_html=True)
            info = INFO_FLORES.get(clase)
            if info:
                st.markdown(f"""
                <div class='info-box'>
                    <strong>🌿 Nombre científico:</strong> <i>{info['nombre_cientifico']}</i><br><br>
                    <strong>📍 Origen:</strong> {info['origen']}<br><br>
                    <strong>🔬 Usos comunes:</strong> {info['uso']}<br><br>
                    <strong>💡 Curiosidad:</strong> {info['curiosidad']}<br><br>
                    <a href="https://www.google.com/search?q=flor+{clase}" target="_blank">🔎 Buscar más en Google</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("🚫 Flor no registrada en la base de datos.")

    with col_img:
        st.image(imagen, caption="📷 Imagen cargada", use_container_width=True)
