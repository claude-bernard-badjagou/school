# Importer les librairies
import pandas as pd
import streamlit as st
import pickle as pk


#Titre de la page
st.title("Prendre des décisions basées sur des données")


# Chargement des données prétraitées dans data manipulation
df = pk.load(open("data_preproccessing.pkl", "rb"))


# Question 1
st.write("**1. Quel est le pourcentage d'étudiant admis dans chaque pays ?**")
admis_par_pays = df[df["admis"]== "oui"].groupby("country").size()
n_etudiant_pays = df.groupby("country").size()
admis_percent = (admis_par_pays / n_etudiant_pays)*100
st.write("Le pourcentage d'étudiant admis dans chaque pays est: \n", pd.DataFrame(admis_percent).T)

# Question 2
st.write("**2. Quel est le nombre de mentions par type obtenues par étudiants dans chaque pays ?**")
n_mention_pays = df.groupby(["country", "mention"]).size().reset_index(name="nombre")
pivot_n_mention_pays = pd.pivot_table(n_mention_pays, values="nombre", index="country", columns="mention", aggfunc="sum")
st.write("Le nombre de mentions par type obtenues par étudiants dans chaque pays", pivot_n_mention_pays)

# Question 3
st.write("**3. Quel est le pourcentage d'étudiant n'ayant pas la moyenne en 'data_modelisation' parmi ceux qui ont échoué ?**")
non_admis_data_mode = (df[df["admis"]== "non"]["data_modelisation"] < 50).mean()*100
st.write("Le pourcentage d'étudiant n'ayant pas la moyenne en 'data_modelisation' "
         "parmi ceux qui ont échoué est", non_admis_data_mode)

# Question 4
st.write("**4. Quel est la moyenne des notes en python des étudiants qui ont obtenus la mention 'assez bien' en 2020 ?**")
moy_python_assez_bien = df[(df["mention"] == "assez bien") & (df["année"] == 2020)]["python"].mean()
st.write("La moyenne des notes en python des étudiants qui ont obtenus la mention 'assez bien' en 2020 est : ", moy_python_assez_bien)

# Question 5
st.write("**5. Quelles sont les trois matières dans lesquelles il faut beaucoup travailler pour réussir facilement**")
# sélectionner les matières avec notes d'évalution
colonne_numeric = df.select_dtypes(include=["int64", "float64"])
# calculer la correlation de chaque colonne numérique avec la moyenne
corr_col_moy = colonne_numeric.corr()["moyenne"].drop("moyenne")
st.write("Voici un tableau du coef de correlation des colonnes numériques avec la moyenne")
st.write(corr_col_moy)
st.write("Les trois matières sont : data_transformation, python, data_manipulation")

# Lien vers les autres pages ou sections
st.subheader("Liens vers les autres pages ou sections")

# Liste des liens
st.write("""
- [Accueil](http://localhost:8501/)
- [Information](http://localhost:8501/Informations)
- [Data Exploration](http://localhost:8501/Data_Exploration)
- [Data Manipulation](http://localhost:8501/Data_Manipulation)
- [Data Visualisation](http://localhost:8501/Data_Visualization)
- [Make Visualisation](http://localhost:8501/Make_Visualization)é
- [Data Modelisation](http://localhost:8501/Data_Modelisation)
- [Contact Us](http://localhost:8501/Contact_Us)
- [About Us](http://localhost:8501/About_Us)
""")



