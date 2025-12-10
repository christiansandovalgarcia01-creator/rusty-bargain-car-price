"""
Rusty Bargain - Car Price Prediction
------------------------------------

Código del proyecto de predicción de precios de autos usados
(desarrollado en plataforma de bootcamp, dataset no incluido
por ser material propietario).

El script muestra:
- Preparación de datos (numéricos y categóricos)
- Definición de un preprocesador con ColumnTransformer
- Entrenamiento y comparación de:
    * Regresión lineal
    * Bosque aleatorio
    * LightGBM

Para ejecutarlo haría falta el archivo 'car_data.csv' en la carpeta /datasets.
"""

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


def prepare_data(path: str = "/datasets/car_data.csv"):
    """Carga y prepara el dataset. El CSV no se incluye en el repo."""
    data = pd.read_csv(path)

    # Eliminar duplicados exactos
    data = data.drop_duplicates()

    # Eliminar NumberOfPictures si es constante
    if "NumberOfPictures" in data.columns and data["NumberOfPictures"].nunique() == 1:
        data = data.drop("NumberOfPictures", axis=1)

    # Separar objetivo y características
    target = data["Price"]
    features = data.drop("Price", axis=1)

    # Split train/valid
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    # Columnas numéricas y categóricas
    numeric_features = features_train.select_dtypes(exclude="object").columns
    categorical_features = features_train.select_dtypes(include="object").columns

    # Preprocesador para modelos clásicos
    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Versión de datos para LightGBM (categorías nativas)
    features_lgbm = features.copy()
    cat_cols_lgbm = features_lgbm.select_dtypes(include="object").columns
    for col in cat_cols_lgbm:
        features_lgbm[col] = features_lgbm[col].astype("category")

    num_cols_lgbm = features_lgbm.select_dtypes(exclude="category").columns
    features_lgbm[num_cols_lgbm] = features_lgbm[num_cols_lgbm].fillna(
        features_lgbm[num_cols_lgbm].median()
    )

    features_train_lgbm, features_valid_lgbm, target_train_lgbm, target_valid_lgbm = train_test_split(
        features_lgbm, target, test_size=0.25, random_state=12345
    )

    return (
        features_train,
        features_valid,
        target_train,
        target_valid,
        preprocessor,
        features_train_lgbm,
        features_valid_lgbm,
        target_train_lgbm,
        target_valid_lgbm,
    )


def train_and_evaluate(model, model_name,
                       features_train, features_valid,
                       target_train, target_valid):
    """Entrena un modelo, mide tiempos y calcula RMSE."""
    start_train = time.time()
    model.fit(features_train, target_train)
    end_train = time.time()

    start_pred = time.time()
    predictions = model.predict(features_valid)
    end_pred = time.time()

    rmse = mean_squared_error(target_valid, predictions, squared=False)
    train_time = end_train - start_train
    pred_time = end_pred - start_pred

    print(f"{model_name}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  Tiempo de entrenamiento: {train_time:.2f} s")
    print(f"  Tiempo de predicción:   {pred_time:.4f} s\n")

    return rmse, train_time, pred_time


def main():
    (
        features_train,
        features_valid,
        target_train,
        target_valid,
        preprocessor,
        features_train_lgbm,
        features_valid_lgbm,
        target_train_lgbm,
        target_valid_lgbm,
    ) = prepare_data()

    # 1) Regresión lineal (baseline)
    linear_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    rmse_lin, ttrain_lin, tpred_lin = train_and_evaluate(
        linear_model,
        "Regresión lineal",
        features_train,
        features_valid,
        target_train,
        target_valid,
    )

    # 2) Bosque aleatorio
    rf_regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=12345,
        n_jobs=-1,
    )

    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", rf_regressor),
        ]
    )

    rmse_rf, ttrain_rf, tpred_rf = train_and_evaluate(
        rf_model,
        "Bosque aleatorio",
        features_train,
        features_valid,
        target_train,
        target_valid,
    )

    # 3) LightGBM
    lgbm_model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=12345,
    )

    rmse_lgbm, ttrain_lgbm, tpred_lgbm = train_and_evaluate(
        lgbm_model,
        "LightGBM",
        features_train_lgbm,
        features_valid_lgbm,
        target_train_lgbm,
        target_valid_lgbm,
    )

    # Resumen simple
    results = pd.DataFrame({
        "Modelo": ["Regresión lineal", "Bosque aleatorio", "LightGBM"],
        "RMSE": [rmse_lin, rmse_rf, rmse_lgbm],
        "Tiempo_entrenamiento_s": [ttrain_lin, ttrain_rf, ttrain_lgbm],
        "Tiempo_prediccion_s": [tpred_lin, tpred_rf, tpred_lgbm],
    })

    print("Resumen de resultados:")
    print(results.sort_values("RMSE"))


if __name__ == "__main__":
    # Sin el CSV, este script sirve como referencia de código.
    main()
