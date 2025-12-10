# Rusty Bargain – Predicción de precios de autos usados

Proyecto de Machine Learning para estimar el precio de autos usados de un servicio tipo **Rusty Bargain**.  
El trabajo se realizó como parte del bootcamp de Data Science (TripleTen).

> ⚠️ El dataset original fue proporcionado por la plataforma y **no puede publicarse**.  
> Por esa razón, este repositorio contiene únicamente el **código del pipeline de modelado**, sin los datos.

## Objetivo

Construir y comparar modelos de regresión que predigan el precio (`Price`) en euros a partir de:

- características técnicas (potencia, kilometraje, año de registro, etc.),
- tipo de carrocería, combustible, marca, modelo,
- información de mantenimiento/reparaciones.

Se evaluó:

- la **calidad del modelo** (métrica RMSE),
- el **tiempo de entrenamiento**,
- la **velocidad de predicción**, pensando en una app que debe responder casi en tiempo real.

## Metodología

1. **Preparación de datos**
   - Eliminación de duplicados.
   - Manejo de valores ausentes.
   - Separación en train/valid (`train_test_split`).
   - Distinción entre variables numéricas y categóricas.

2. **Preprocesamiento**
   - Modelos clásicos:
     - `SimpleImputer` (mediana para numéricas, moda para categóricas).
     - `OneHotEncoder` para variables categóricas.
     - Integración con `ColumnTransformer` y `Pipeline`.
   - LightGBM:
     - Conversión de columnas categóricas a tipo `category` de pandas.
     - Aprovechando el soporte nativo de LightGBM para estas variables.

3. **Modelos entrenados**
   - `LinearRegression` – modelo baseline / prueba de cordura.
   - `RandomForestRegressor` – modelo de árboles en conjunto.
   - `LGBMRegressor` – implementación de gradient boosting sobre árboles.

4. **Comparación**
   - Métrica principal: **RMSE (Root Mean Squared Error)**.
   - Medición de:
     - tiempo de entrenamiento,
     - tiempo de predicción,
   con la librería estándar `time`.

## Resultados (resumen cualitativo)

- La **regresión lineal** sirve como referencia, pero no captura bien la relación no lineal entre variables y precio.
- El **bosque aleatorio** mejora la métrica, a costa de un mayor tiempo de entrenamiento.
- **LightGBM** ofrece el mejor equilibrio entre:
  - error más bajo (mejor RMSE),
  - entrenamiento razonablemente rápido,
  - predicciones muy rápidas para uso en una app.

Por estos motivos, **LightGBM** se considera el modelo principal recomendado para integrar en una futura aplicación de Rusty Bargain.

## Tecnologías

- Python
- pandas, NumPy
- scikit-learn
- LightGBM

## Código

El pipeline completo se encuentra en:

- `src/rusty_bargain_car_price.py`

El script está organizado en funciones:

- `prepare_data()` – preparación de datos y preprocesador,
- `train_and_evaluate()` – entrenamiento y evaluación de cada modelo,
- `main()` – orquestación de todo el flujo.

> Nota: Para ejecutar el script se requeriría el archivo `car_data.csv` original, que no se incluye en este repositorio.

