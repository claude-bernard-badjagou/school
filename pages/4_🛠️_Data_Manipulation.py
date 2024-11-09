# Importer les librairies
import streamlit as st
import pandas as pd
from datetime import datetime

#Titre de la page
st.title("Traitement des données")


# Chargement de données du fichier csv
df = pd.read_csv("data.csv")


# Remplaçons les notes manquantes en Excel, tableau et pack office par 0
df[["excel", "tableau", "pack_office"]] = df[["excel", "tableau", "pack_office"]].fillna(0)
st.write("1. Les notes manquantes des colonnes excel, tableau et pack office sont rémplacées par 0")


# Convertir le type de données de la colonne date en datetime
df["date"] = pd.to_datetime(df["date"])
st.success("2. Le type object de la colonne date est converti en datetime")



"""Remplacement des valeurs manquantes par le mode ou la mediane selon le type de données
for col in df.columns:

    if df[col].dtypes == "object":
        mode_col = df[col].mode()[0]
        df[col].fillna(mode_col, inplace=True)
    
    elif df[col].dtypes in ["int64", "float64"]:
        mediane_col = df[col].median()
        df[col] = df[col].fillna(mediane_col)
    
    else:
        pass"""


# Suppression des valeurs dupliquées
df = df.drop_duplicates()
st.error("3. Suppression des valeurs dupliquées")


# Gestion des valeurs aberantes
df.loc[15, "data_manipulation"] = 50
df.loc[1506, "data_modelisation"] = 50
st.warning("4. Remplacement des valeurs aberantes par la moyenne 50")

# Création de nouvelles caractéristiques (colonnes)
# pour de l'analyse de données, de la visualisation et des models


# colonne moyenne
df["moyenne"]= (df["sql"]*2 + df["excel"]*0.05 + df["maths"]*4 + df["python"]*5 + df["poo"]*3 +
               df["tableau"]*2 + df["data_exploration"]*2 + df["data_manipulation"]*5 +
               df["data_viz"]*4 + df["data_transformation"]*5 + df["data_modelisation"]*6 +
               df["data_deployement"]*5 + df["pack_office"]*2 + df["result_presentation"]*5) / 50
st.success("La colonne moyenne a été ajouté")

# colonne admis
df["admis"] = df["moyenne"].apply(lambda moyenne: "oui" if moyenne>=50 else "non")
st.success("La colonne moyenne a été ajouté")

# colonne mention
df["mention"] = df["moyenne"].apply(lambda moyenne:
                                    "refusée" if moyenne<50  else
                                    "passable" if moyenne<60  else
                                    "assez bien" if moyenne<70  else
                                    "bien" if moyenne<80  else
                                    "très bien" if moyenne<90  else
                                    "excellente")
st.success("La colonne mention a été ajouté")

# colonne année
df["année"] = df["date"].dt.year
st.success("La colonne année a été ajouté")


# Enregistrement des données au format picle (pkl)
df.to_pickle("data_preproccessing.pkl")
st.success("Les données ont été enregistré au format picle afin de l'utiliser pour la suite")


# Lien vers les autres pages ou sections
st.subheader("Liens vers les autres pages ou sections")

# Liste des liens
st.write("""
- [Accueil](http://localhost:8501/)
- [Information](http://localhost:8501/Informations)
- [Data Exploration](http://localhost:8501/Data_Exploration)
- [Driven Décision](http://localhost:8501/Driven_decision)
- [Data Visualisation](http://localhost:8501/Data_Visualization)
- [Make Visualisation](http://localhost:8501/Make_Visualization)é
- [Data Modelisation](http://localhost:8501/Data_Modelisation)
- [Contact Us](http://localhost:8501/Contact_Us)
- [About Us](http://localhost:8501/About_Us)
""")
