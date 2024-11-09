# Importer les librairies
import streamlit as st
import pandas as pd

#Titre de la page
st.title("Information sur ces données")


# Chargement de données du fichier csv
df = pd.read_csv("data.csv")


# Afficher les données
st.warning("Voici les données fictives générées pour réaliser cette application")
st.write(df)

# Afficher les noms des colonnes
st.subheader("Information sur les noms des colonnes")
st.write("""
**student_id :** Contient les identifiants uniques des étudiants

**date :** Contient la date d'obtention du diplôme

**name :** Contient les noms des étudiants de l'école

**country :** Contient les pays d'origine des étudiants

**Les colonnes : sql, excel, maths, python, poo, tableau, data_exploration, data_manipulation, data_viz
data_transformation, data_modelisation, data_deployement, pack_office, result_presentation contiennent les 
notes d'évaluation.
""")


# Lien vers les autres pages ou sections
st.subheader("Liens vers les autres pages ou sections")

# Liste des liens
st.write("""
- [Accueil](http://localhost:8501/)
- [Data_Exploration](http://localhost:8501/Data_Exploration)
- [Data Manipulation](http://localhost:8501/Data_Manipulation)
- [Driven Décision](http://localhost:8501/Driven_decision)
- [Data Visualisation](http://localhost:8501/Data_Visualization)
- [Make Visualisation](http://localhost:8501/Make_Visualization)
- [Data Modelisation](http://localhost:8501/Data_Modelisation)
- [Contact Us](http://localhost:8501/Contact_Us)
- [About Us](http://localhost:8501/About_Us)
""")
