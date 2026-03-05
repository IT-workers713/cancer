import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configuration de la page
st.set_page_config(page_title="Détection de Cancer - Computer Vision", layout="wide")

# -------------------------------------------------------------------
# 1. Chargement du modèle avec mise en cache
# -------------------------------------------------------------------
@st.cache_resource
def load_my_model():
    model_path = "best_model_MobileNetV2.keras"
    return tf.keras.models.load_model(model_path)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# -------------------------------------------------------------------
# 2. Détermination automatique du nom de la dernière couche convolutive
# -------------------------------------------------------------------
def get_last_conv_layer(model):
    """Parcourt le modèle à l'envers et retourne le nom de la première couche Conv2D trouvée."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Aucune couche Conv2D trouvée dans le modèle.")

LAST_CONV_LAYER = get_last_conv_layer(model)

# -------------------------------------------------------------------
# 3. Fonction Grad-CAM (adaptée pour cibler la classe CANCER)
# -------------------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, target_class='cancer'):
    """
    Génère une heatmap Grad-CAM pour l'image donnée.
    - target_class : 'cancer' ou 'non_cancer'
    """
    # Modèle qui donne à la fois les sorties de la couche conv et la prédiction
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        outputs = grad_model(img_array)          # [conv_outputs, predictions]
        conv_outputs = outputs[0]
        predictions = outputs[1]

        # Si predictions est une liste (plusieurs sorties), on prend la première
        if isinstance(predictions, list):
            predictions = predictions[0]

        # La sortie est un scalaire (sigmoid) : probabilité de NON-CANCER
        prob_non_cancer = predictions[:, 0]

        if target_class == 'cancer':
            loss = 1.0 - prob_non_cancer   # gradients pour la classe cancer
        else:
            loss = prob_non_cancer          # gradients pour la classe non-cancer

    # Calcul des gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Pondération des cartes d'activation
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # ReLU et normalisation
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    return heatmap.numpy()

# -------------------------------------------------------------------
# 4. Fonction pour extraire le rectangle englobant de la zone la plus chaude
# -------------------------------------------------------------------
def get_bounding_box_from_heatmap(heatmap, threshold=0.5):
    """
    Extrait le plus grand rectangle englobant à partir d'une heatmap seuillée.
    Retourne (x, y, w, h) ou None si aucun contour.
    """
    # Seuillage binaire
    binary = np.uint8(heatmap > threshold) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Prendre le plus grand contour (par surface)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return x, y, w, h

# -------------------------------------------------------------------
# 5. Interface utilisateur
# -------------------------------------------------------------------
st.title("🩺 Analyse d'Images Médicales - Détection de Cancer Oral")
st.write("Téléchargez une image pour identifier les zones suspectes et obtenir une probabilité.")

uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lecture et prétraitement de l'image
    img = Image.open(uploaded_file).convert('RGB')
    img_resized = img.resize((224, 224))

    # Préparation pour le modèle (prétraitement MobileNetV2)
    img_array = np.array(img_resized).astype(np.float32)
    img_array = preprocess_input(img_array)          # mise à l'échelle [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)    # (1, 224, 224, 3)

    # Prédiction
    preds = model.predict(img_array, verbose=0)
    raw_prob = preds[0][0]          # probabilité pour la classe 1 (NON-CANCER)

    # Interprétation
    prob_cancer = 1 - raw_prob
    prob_non_cancer = raw_prob
    resultat = "CANCER" if raw_prob < 0.5 else "NON-CANCER"

    # Affichage sur deux colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image Originale")
        st.image(img, use_container_width=True)

        st.metric(label="Résultat Final", value=resultat)
        st.write(f"**Probabilité Cancer :** {prob_cancer:.2%}")
        st.write(f"**Probabilité Sain :** {prob_non_cancer:.2%}")

    with col2:
        st.subheader("Localisation des zones suspectes")

        # Génération de la heatmap pour la classe CANCER
        heatmap_cancer = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER, target_class='cancer')

        # Préparation de l'image pour OpenCV
        img_cv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        heatmap_resized = cv2.resize(heatmap_cancer, (img_cv.shape[1], img_cv.shape[0]))

        # Création de la superposition classique (heatmap rouge)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

        # Extraction du rectangle englobant
        bbox = get_bounding_box_from_heatmap(heatmap_resized, threshold=0.5)

        if bbox is not None:
            x, y, w, h = bbox
            # Dessiner un cadre vert sur l'image superposée
            cv2.rectangle(superimposed, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Affichage dans Streamlit
        st.image(superimposed, channels="BGR", use_container_width=True)
        st.caption("Les zones rouges indiquent les régions influentes ; le cadre vert délimite la zone suspecte.")