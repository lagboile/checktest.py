import pandas as pd

# Charger les données
df = pd.read_csv("iris.csv")

# Exploration de base
print(df.head())  # Premières lignes
print(df.info())  # Informations générales
print(df.describe())  # Statistiques de base

from ydata_profiling import ProfileReport

# Génération du rapport
profile = ProfileReport(df, title="Rapport de Profiling", explorative=True)
profile.to_file("rapport_profiling.html")

# Identifier les valeurs manquantes
print(df.isnull().sum())

# Remplir ou supprimer les valeurs manquantes

df.dropna(inplace=True)  # Suppression des lignes avec valeurs manquantes

# Identifier les doublons
print(f"Doublons : {df.duplicated().sum()}")

# Supprimer les doublons
df.drop_duplicates(inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt

# Boîte à moustaches pour visualiser les aberrations
sns.boxplot(data=df.select_dtypes(include=["float", "int"]))
plt.show()

# Identifier les colonnes numériques
numeric_columns = df.select_dtypes(include=["number"]).columns

# Appliquer quantile uniquement sur les colonnes numériques
q1 = df[numeric_columns].quantile(0.25)
q3 = df[numeric_columns].quantile(0.75)
iqr = q3 - q1  # Intervalle interquartile

print("Q1 :", q1)
print("Q3 :", q3)
print("IQR :", iqr)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Diviser les données
X = df.drop("species", axis=1)  # Remplacez "cible" par votre colonne cible
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Former le modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Tester le modèle
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

import streamlit as st
import numpy as np

# Charger le modèle
st.title("Application de Classification")
st.write("Remplissez les champs pour obtenir une prédiction.")

# Créer des champs de saisie
features = []
for col in X.columns:
    val = st.number_input(f"Valeur pour {col}", min_value=float(X[col].min()), max_value=float(X[col].max()))
    features.append(val)

# Prédire lorsque le bouton est cliqué
if st.button("Prédire"):
    prediction = model.predict([features])
    st.write(f"Prédiction : {prediction[0]}")
