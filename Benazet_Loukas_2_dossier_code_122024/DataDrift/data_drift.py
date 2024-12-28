import evidently
import time
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
import pandas as pd 
import numpy as np

df = pd.read_csv('../Notebooks/df.csv', sep = ",")
application_train = df.dropna(subset=['TARGET']).drop(columns=['SK_ID_CURR','TARGET'])
application_test = df[df['TARGET'].isna()].drop(columns=['SK_ID_CURR','TARGET'])

#Pour les colonnes catégorielles on va ne prendre que les colonnes ayant que des 0 et des 1 
categorical_columns = []

# Parcourir chaque colonne
for col in application_train.columns:
    # Récupérer les valeurs uniques de la colonne
    unique_vals = set(application_train[col].unique())
    
    # Vérifier si les valeurs uniques sont uniquement 0, 1, et potentiellement NaN
    if unique_vals.issubset({0, 1, np.nan}):
        categorical_columns.append(col)

numerical_columns = [col for col in application_train.columns if col not in categorical_columns]

start_time = time.time()

# Vérifier que vos deux DataFrames ont exactement les mêmes colonnes
assert set(application_train.columns) == set(application_test.columns)

# Si l'assertion est réussie, cela signifie que les colonnes correspondent
print("Les colonnes correspondent!")

# Création du column mapping
column_mapping = ColumnMapping()

column_mapping.numerical_features = numerical_columns
column_mapping.categorical_features = categorical_columns

# Créer le rapport de dérive des données
data_drift_report = Report(metrics=[
    DataDriftPreset(num_stattest='ks', cat_stattest='psi', num_stattest_threshold=0.2, cat_stattest_threshold=0.2),
])

print("Création du data_drift_report")

data_drift_report.run(reference_data=application_train, current_data=application_test, column_mapping=column_mapping)

print("Run du data_drift_report")

elapsed_time_fit = time.time() - start_time
print(elapsed_time_fit)

# Sauvegarder le rapport en tant que fichier HTML
data_drift_report.save_html('data_drift_report_FULL_script.html')
