import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Détection de Cancer de la Peau",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 1.2rem;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Charge le modèle et les informations"""
    try:
        # Charger les informations du modèle
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        num_classes = len(model_info['class_names'])
        
        # Créer la classe du modèle comme dans le notebook
        class ResNet50Model(nn.Module):
            def __init__(self, num_classes):
                super(ResNet50Model, self).__init__()
                self.resnet = models.resnet50(weights=None)
                in_features = self.resnet.fc.in_features
                # Structure exacte du modèle sauvegardé (indices 1, 2, 5)
                self.resnet.fc = nn.Sequential(
                    nn.Dropout(0.5),           # index 0
                    nn.Linear(in_features, 512),  # index 1
                    nn.BatchNorm1d(512),       # index 2
                    nn.ReLU(),                 # index 3
                    nn.Dropout(0.5),           # index 4
                    nn.Linear(512, num_classes)   # index 5
                )
            
            def forward(self, x):
                return self.resnet(x)
        
        # Créer le modèle
        model = ResNet50Model(num_classes)
        
        # Charger les poids
        state_dict = torch.load('best_model.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, model_info
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return None, None

def preprocess_image(image, model_info):
    """Prétraite l'image pour le modèle"""
    # Définir les transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=model_info['mean'],
            std=model_info['std']
        )
    ])
    
    # Convertir en RGB si nécessaire
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Appliquer les transformations
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)  # Ajouter dimension batch
    
    return img_tensor

def predict(model, image_tensor, model_info):
    """Effectue la prédiction"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item(), probabilities[0].numpy()

def get_class_info(class_name):
    """Retourne des informations détaillées sur chaque classe"""
    info = {
        "akiec": {
            "nom": "Kératose actinique",
            "description": "Lésion précancéreuse causée par l'exposition au soleil",
            "gravite": "Modérée",
            "couleur": "#FFA500"
        },
        "bcc": {
            "nom": "Carcinome basocellulaire",
            "description": "Type de cancer de la peau le plus courant, généralement non mortel",
            "gravite": "Élevée",
            "couleur": "#FF6347"
        },
        "bkl": {
            "nom": "Kératose bénigne",
            "description": "Lésion cutanée bénigne, non cancéreuse",
            "gravite": "Faible",
            "couleur": "#90EE90"
        },
        "df": {
            "nom": "Dermatofibrome",
            "description": "Tumeur bénigne du tissu conjonctif de la peau",
            "gravite": "Faible",
            "couleur": "#87CEEB"
        },
        "mel": {
            "nom": "Mélanome",
            "description": "Forme la plus dangereuse de cancer de la peau",
            "gravite": "Très élevée",
            "couleur": "#DC143C"
        },
        "nv": {
            "nom": "Nævus mélanocytaire",
            "description": "Grain de beauté commun, généralement bénin",
            "gravite": "Faible",
            "couleur": "#98FB98"
        },
        "vasc": {
            "nom": "Lésions vasculaires",
            "description": "Anomalies des vaisseaux sanguins de la peau",
            "gravite": "Faible à modérée",
            "couleur": "#DDA0DD"
        }
    }
    return info.get(class_name, {})

def create_probability_chart(probabilities, class_names, class_labels_fr):
    """Crée un graphique des probabilités"""
    # Préparer les données
    labels = [class_labels_fr[name] for name in class_names]
    probs = [prob * 100 for prob in probabilities]
    colors = [get_class_info(name)['couleur'] for name in class_names]
    
    # Créer le graphique à barres
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=labels,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{p:.1f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Probabilités pour chaque classe",
        xaxis_title="Probabilité (%)",
        yaxis_title="Type de lésion",
        height=400,
        showlegend=False
    )
    
    return fig

# Interface principale
def main():
    # En-tête
    st.title("🔬 Système de Détection de Cancer de la Peau")
    st.markdown("### Application d'analyse d'images dermatologiques basée sur l'IA")
    
    # Charger le modèle
    model, model_info = load_model()
    
    if model is None:
        st.error("⚠️ Impossible de charger le modèle. Vérifiez que les fichiers 'best_model.pth' et 'model_info.pkl' sont présents.")
        return
    
    # Sidebar avec informations
    with st.sidebar:
        st.header("ℹ️ Informations sur le modèle")
        st.write(f"**Architecture:** {model_info['model_name']}")
        st.write(f"**Précision de validation:** {float(model_info['best_val_accuracy']):.2f}%")
        st.write(f"**Précision de test:** {float(model_info['test_accuracy']):.2f}%")
        st.write(f"**Nombre de classes:** {len(model_info['class_names'])}")
        
        st.markdown("---")
        st.header("📋 Classes détectées")
        for class_name in model_info['class_names']:
            info = get_class_info(class_name)
            st.markdown(f"**{info['nom']}** ({class_name})")
            st.caption(info['description'])
            st.markdown("")
    
    # Zone d'avertissement
    st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Avertissement médical:</strong> Cette application est un outil d'aide à la décision 
            et ne remplace pas un diagnostic médical professionnel. Consultez toujours un dermatologue 
            pour un diagnostic définitif.
        </div>
    """, unsafe_allow_html=True)
    
    # Zone de téléchargement d'image
    st.header("📤 Télécharger une image")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choisissez une image de lésion cutanée",
            type=['jpg', 'jpeg', 'png'],
            help="Formats acceptés: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Afficher l'image
            image = Image.open(uploaded_file)
            st.image(image, caption="Image téléchargée", use_container_width=True)
            
            # Bouton de prédiction
            if st.button("🔍 Analyser l'image"):
                with st.spinner("Analyse en cours..."):
                    # Prétraiter l'image
                    img_tensor = preprocess_image(image, model_info)
                    
                    # Faire la prédiction
                    predicted_idx, confidence, probabilities = predict(model, img_tensor, model_info)
                    
                    # Obtenir le nom de la classe
                    predicted_class = model_info['class_names'][predicted_idx]
                    predicted_label = model_info['class_labels_fr'][predicted_class]
                    class_info = get_class_info(predicted_class)
                    
                    # Afficher les résultats dans la colonne 2
                    with col2:
                        st.markdown("### 📊 Résultats de l'analyse")
                        
                        # Boîte de prédiction principale
                        st.markdown(f"""
                            <div class="prediction-box" style="border-left: 5px solid {class_info['couleur']}">
                                <h2 style="color: {class_info['couleur']}; margin-top: 0;">
                                    {predicted_label}
                                </h2>
                                <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                                    <strong>Confiance:</strong> {confidence * 100:.2f}%
                                </p>
                                <p style="margin: 0.5rem 0;">
                                    <strong>Description:</strong> {class_info['description']}
                                </p>
                                <p style="margin: 0.5rem 0;">
                                    <strong>Niveau de gravité:</strong> {class_info['gravite']}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Recommandations
                        if class_info['gravite'] in ["Élevée", "Très élevée"]:
                            st.error("⚠️ **Recommandation:** Consultez un dermatologue rapidement.")
                        elif class_info['gravite'] == "Modérée":
                            st.warning("⚠️ **Recommandation:** Consultez un dermatologue pour un suivi.")
                        else:
                            st.info("ℹ️ **Recommandation:** Surveillez l'évolution et consultez en cas de changement.")
                    
                    # Graphique des probabilités (pleine largeur)
                    st.markdown("---")
                    st.markdown("### 📈 Distribution des probabilités")
                    fig = create_probability_chart(
                        probabilities,
                        model_info['class_names'],
                        model_info['class_labels_fr']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau détaillé
                    st.markdown("### 📋 Détails des probabilités")
                    results_data = []
                    for i, class_name in enumerate(model_info['class_names']):
                        results_data.append({
                            "Classe": model_info['class_labels_fr'][class_name],
                            "Code": class_name,
                            "Probabilité (%)": f"{probabilities[i] * 100:.2f}%",
                            "Gravité": get_class_info(class_name)['gravite']
                        })
                    
                    st.dataframe(results_data, use_container_width=True)
    
    with col2:
        if uploaded_file is None:
            st.info("👈 Téléchargez une image pour commencer l'analyse")
            
            # Exemple d'utilisation
            st.markdown("### 💡 Comment utiliser cette application")
            st.markdown("""
                1. **Téléchargez** une image de lésion cutanée
                2. **Cliquez** sur le bouton "Analyser l'image"
                3. **Consultez** les résultats et recommandations
                4. **Partagez** les résultats avec votre médecin
            """)
            
            st.markdown("### 📸 Conseils pour de meilleures photos")
            st.markdown("""
                - Utilisez un bon éclairage naturel
                - Prenez la photo à une distance de 10-15 cm
                - Assurez-vous que la lésion est au centre
                - Évitez les ombres et les reflets
                - Utilisez un fond neutre si possible
            """)

if __name__ == "__main__":
    main()
