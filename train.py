import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
import pickle
from azureml.core import Workspace, Experiment
from azureml.core.run import Run

# # Connect to Azure Machine Learning workspace
# workspace = Workspace.from_config()  # Assuming you have a config file for your workspace

# # Start an experiment
# experiment = Experiment(workspace, "car-price-prediction")

# Get the current run
run = Run.get_context()

df = pd.read_csv('prix_voiture.csv')

# Define X and y variables
y = df['prix']
X = df[['taille_moteur', 'poids_vehicule', 'chevaux', 'consommation_autoroute', 'consommation_ville', 'largeur', 'longueur', 'roues_motrices', 'empattement', 'systeme_carburant', 'turbo', 'marque', 'etat_de_route', 'type_vehicule', 'emplacement_moteur', 'type_moteur', 'nombre_cylindres']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

categorical_features = ['etat_de_route', 'turbo', 'type_vehicule', 'roues_motrices', 'emplacement_moteur', 'type_moteur', 'nombre_cylindres', 'systeme_carburant', 'marque']
numeric_features = ['empattement', 'longueur', 'largeur', 'poids_vehicule', 'taille_moteur', 'chevaux', 'consommation_ville', 'consommation_autoroute']

categorical_transformer = Pipeline(steps=[
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])
numeric_transformer = Pipeline([
    ('min_max', MinMaxScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

rf_model = Pipeline([
    ('preprocess', preprocessor),
    ('random_forest', RandomForestRegressor(n_estimators=15, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42))
])

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_model.predict(X_test)

# Calculate the R^2 score on the test data
score = rf_model.score(X_test, y_test)

# Log the R^2 score
run.log("R^2 score", score)

# Save the trained model
model_file = 'random_forest.pkl'
with open(model_file, 'wb') as file:
    pickle.dump(rf_model, file)

# Register the trained model in Azure Machine Learning workspace
run.upload_file('outputs/' + model_file, model_file)
model = run.register_model(model_name='car_price_model', model_path='outputs/' + model_file)

# Store any additional results or artifacts in Azure storage
# ...

# Complete the run
run.complete()
