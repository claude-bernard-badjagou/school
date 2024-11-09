# Importer les librairies
import streamlit as st
import pandas as pd

#Titre de la page
st.title("Exploration des données")


# Chargement de données du fichier csv
df = pd.read_csv("data.csv")
df = pd.DataFrame(df)

# Information sur les données

# Nombre de ligne et de colonne dans le dataframe
st.write("**Forme du dataframe**")
n_ligne, n_colonne = df.shape
st.write(f"""
\n Le nombre de ligne dans le dataframe est : {n_ligne}
\n Le nombre de colonnne dans le dataframe est : {n_colonne}
""")


# Vérification des types de données
st.info("Les types de données par colonne")
for col in df.columns:
    dt = df[col].dtypes
    st.write(f"Le type de donnée de la colonne **{col}** est **{dt}**")


# Vérification des valeurs manquantes
valeurs_manquantes = df.isnull().sum()
st.warning("Valeurs manquantes par colonne")
st.write(pd.DataFrame(valeurs_manquantes).T)


# Vérification des valeurs dupliquées
st.error("**Valeurs dupliquées**")
duplicated = df[df.duplicated()]
st.write("Données dupliquées", duplicated)


# Vérification des valeurs aberantes
st.info("Valeurs aberantes")
# Seuil pour idenfier une valeur aberante en fonction de z-score
seuil = 3

# Parcourir chaque colonne numérique pour calculer la moyenne et l'écart type
colonne_numeric = df.select_dtypes(include=['int64', 'float64']).columns
for colonne in colonne_numeric:
    # calculer la moyenne de la colonne actuelle
    moyenne_col = df[colonne].mean()
    # calculer l'écart type de la colonne actuelle
    ecart_type_col = df[colonne].std()
    # Calculer le z_score de chaque ligne de la colonne
    z_score = (df[colonne] - moyenne_col) / ecart_type_col

    # Identifier les valeurs aberantes selon la valeur de z_score
    valeur_aberante = df[ abs(z_score) > seuil]

    # Afficher les valeurs aberantes
    if not valeur_aberante.empty:
        st.write(f"Il y a valeur aberante dans la colonne '{colonne}' :")
        for index, valeur in valeur_aberante[colonne].items():
            st.write(f"La valeur aberante se trouve à la ligne {index} et sa valeur est {valeur}")

# Staistique descriptive
st.write(" **Statistique descriptive sur les colonnes numériques**")
st.write(df.describe())


# Lien vers les autres pages ou sections
st.subheader("Liens vers les autres pages ou sections")

# Liste des liens
st.write("""
- [Accueil](http://localhost:8501/)
- [Information](http://localhost:8501/Informations)
- [Data Manipulation](http://localhost:8501/Data_Manipulation)
- [Driven Décision](http://localhost:8501/Driven_decision)
- [Data Visualisation](http://localhost:8501/Data_Visualization)
- [Make Visualisation](http://localhost:8501/Make_Visualization)
- [Data Modelisation](http://localhost:8501/Data_Modelisation)
- [Contact Us](http://localhost:8501/Contact_Us)
- [About Us](http://localhost:8501/About_Us)
""")
