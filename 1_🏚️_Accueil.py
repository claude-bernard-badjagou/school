# installation des bibliothèques à importer

# Importer les librairies
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title= "School application",
    page_icon= "🎓"
)


# Titre de la page
st.title("Les résultats de mon école")


# Image illustratif de l'application
img = Image.open("gomycode.jpg")
st.image(img, caption= "Image de gomycode sur Facebook", use_column_width=True)


# A propos de l'application
st.write("""
<div style= "text-align: justify">
Cette application est conçue pour suivre les performances des étudiants en data science. 
Leurs performances dans différents domaines en analysant leur résultat pour fournir des 
informations précieuses pour la prise de décision. De plus les modèles sont crés pour prédire
ou classifier les informations sur les étudiants.
</div>
""", unsafe_allow_html=True)

# Fonctionnalité de l'application
st.header("Fonctionnalités de l'application")


# Information sur les données
st.write("**Information:** Information sur les données")


# Exploration des données
st.write("**Data Exploration:** Exploration des données")


# Manipulation des données
st.write("**Data Manipulation:** Manipulation des données")


# Décisions basées sur des données
st.write("**Driven Décision:** Prendre des décisions en se basant sur les données")


# Visualisation des données
st.write("**Data Visualisation:** Visualisation des données")


# Faites vos visualisation des données
st.write("**Make Visualisation:** Faites vos propres visualisations des données")


# Création des données
st.write("**Data Modelisation:** Création des modèles de machine learning et de deep learning")


# Prendre contact
st.write("**Contact Us:** Prendre contact pour plus d'information")


# A propos de nous
st.write("**About Us:** Qui somme nous et comment nous rejoindre")


# Lien vers les autres pages ou sections
st.subheader("Liens vers les autres pages ou sections")

# Liste des liens
st.write("""
- [Information](http://localhost:8501/Informations)
- [Data_Exploration](http://localhost:8501/Data_Exploration)
- [Data Manipulation](http://localhost:8501/Data_Manipulation)
- [Driven Décision](http://localhost:8501/Driven_decision)
- [Data Visualisation](http://localhost:8501/Data_Visualization)
- [Make Visualisation](http://localhost:8501/Make_Visualization)
- [Data Modelisation](http://localhost:8501/Data_Modelisation)
- [Contact Us](http://localhost:8501/Contact_Us)
- [About Us](http://localhost:8501/About_Us)
""")
