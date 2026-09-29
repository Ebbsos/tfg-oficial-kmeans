# Auditoría de la evaluación de estabilidad (ARI y Jaccard)

Descripción de lo que hace exactamente el código de [main.py](main.py), sin modificarlo. Los resultados citados corresponden a la corrida [src/ejecucion_3/](src/ejecucion_3/).

La estabilidad se evalúa en tres funciones:

- `jaccard_por_cluster`: [main.py:500-509](main.py#L500-L509)
- `evaluar_estabilidad`: [main.py:512-534](main.py#L512-L534)
- `validar_estabilidad`: [main.py:537-561](main.py#L537-L561)

---

## 1. Remuestreo

- **Tipo:** remuestreo **sin reemplazo**, es decir, submuestreo (*subsampling*). **No es bootstrap.**
  [main.py:527](main.py#L527): `idx = rng.choice(len(X_final), size=tamano, replace=False)`
- **Fracción:** 80 % (`fraccion=0.8`, [main.py:512](main.py#L512)). `tamano = int(0.8 × 3757) = 3005` clientes ([main.py:525](main.py#L525)).
- **Repeticiones por cada k evaluado:**
  - 20 submuestras (`n_submuestras=20`).
  - 10 reentrenamientos con otras semillas sobre todos los datos (`n_semillas=10`). Esta es una prueba distinta que no remuestrea datos.
- **Semillas:**
  - **Modelo de referencia:** `random_state=42` ([main.py:513](main.py#L513)).
  - **Prueba de semillas:** `random_state` = 1, 2, …, 10 ([main.py:517-518](main.py#L517-L518)).
  - **Prueba de submuestras:**
    - El K-Means de cada submuestra usa siempre `random_state=42` ([main.py:528](main.py#L528)).
    - Los índices de las 20 submuestras se sortean con un único generador `np.random.default_rng(42)` ([main.py:524](main.py#L524)), que produce 20 submuestras distintas en secuencia.
    - Como el generador se crea de nuevo en cada llamada, para k = 2, 3 y 4 se usan las mismas 20 submuestras.

## 2. ARI

- **Función:** `sklearn.metrics.adjusted_rand_score(base, labels)` ([main.py:520](main.py#L520) y [main.py:531](main.py#L531)).
- **Partición de referencia:** `base`, un K-Means con el mismo k, `random_state=42` y `n_init=20`, entrenado sobre **los 3757 clientes** ([main.py:513](main.py#L513)). Coincide con el modelo final del análisis.
- **Procedimiento en cada submuestra:**
  1. Se reajusta K-Means solo con los 3005 clientes sorteados: `.fit(X_final.iloc[idx])`.
  2. Se predice sobre **todos los clientes**, los 3757: `modelo.predict(X_final)`, [main.py:529](main.py#L529).
  3. El ARI compara 3757 etiquetas contra 3757 etiquetas. De esos clientes, 3005 participaron en el entrenamiento y 752 no.

## 3. Jaccard

- **Cálculo:** se hace **por clúster**, por co-pertenencia (intersección sobre unión de los conjuntos de clientes). La función es propia, escrita con NumPy; no se usa sklearn ([main.py:503-508](main.py#L503-L508)):

  ```python
  miembros = labels_base == c
  max(
      (miembros & (labels_nuevas == j)).sum() / (miembros | (labels_nuevas == j)).sum()
      for j in np.unique(labels_nuevas)
  )
  ```

  Fórmula: J(Cᵢ) = max over j of |Cᵢ ∩ C′ⱼ| / |Cᵢ ∪ C′ⱼ|

- **Clientes considerados:** los **3757**, porque ambas particiones cubren a todos gracias al `predict` sobre el conjunto completo. No se restringe a los clientes presentes en la submuestra.
- **Emparejamiento de clústeres:** por **máximo Jaccard**. A cada clúster de la referencia se le asigna el clúster remuestreado con el que más se solapa. No se usa el emparejamiento húngaro ni un matching uno a uno.
- **Umbrales:** **no hay ninguno hardcodeado** (ni 0,85, ni 0,75, ni 0,60). El código solo reporta la media y el mínimo por clúster ([main.py:548-554](main.py#L548-L554)), y únicamente para el k final. Cualquier umbral de interpretación que se cite en la tesis proviene de la literatura, no del código.

## 4. K-Means y selección de k

- **Parámetros:** `n_clusters=k`, `random_state=42`, `n_init=20` (constantes en [main.py:35-36](main.py#L35-L36)). `init` y `max_iter` no se especifican, así que rigen los valores por defecto de scikit-learn ≥ 1.3: `init="k-means++"` y `max_iter=300`.
- **k evaluados:** de 1 a 10 (`K_VALUES`, [main.py:37](main.py#L37)). La silueta se calcula de k = 2 a 10, con todos los clientes (el muestreo solo se activa por encima de 10 000).
- **Criterio de elección en el código:** la mejor silueta (`best_k`, [main.py:344-345](main.py#L344-L345)). El codo se calcula con `KneeLocator` y solo se reporta. El argumento `--k` permite forzar otro valor.
- **Resultados de ejecucion_3** ([metricas_k.csv](src/ejecucion_3/metricas_k.csv)):

  | k | Inercia | Silueta |
  |---|---|---|
  | 1 | 36 548,0 | — |
  | 2 | 16 984,2 | 0,6449 |
  | 3 | 11 238,0 | 0,4297 |
  | 4 | 9 585,0 | 0,3749 |
  | 5 | 8 309,1 | 0,2541 |
  | 6 | 7 259,4 | 0,2691 |
  | 7 | 6 567,7 | 0,2682 |
  | 8 | 6 083,3 | 0,2650 |
  | 9 | 5 614,5 | 0,2687 |
  | 10 | 5 265,2 | 0,2605 |

  - **Codo:** k = 3, según [01_codo.png](src/ejecucion_3/01_codo.png). No se guarda en CSV, solo en la figura y en la salida por consola.
  - **Mejor silueta:** k = 2 (0,645).
  - **Modelo final:** k = 3, con clústeres de 2461, 1023 y 273 clientes. La corrida se hizo con `--k 3`: **k se fijó manualmente en 3, coincidiendo con el codo, y no con la mejor silueta.** La tesis debería decirlo así.
- **k evaluados en la estabilidad:** 2, 3 y 4, más el k final. Como el k final es 3, son solo 2, 3 y 4.

## 5. Resultados (ejecucion_3)

Se guardan en CSV; no hace falta modificar el código para obtenerlos.

### ARI ([estabilidad_resumen.csv](src/ejecucion_3/estabilidad_resumen.csv))

| k | Semillas (n = 10) | Submuestras 80 % (n = 20): media ± desv. | Submuestras: mín. – máx. |
|---|---|---|---|
| 2 | 1,000 | 0,9938 ± 0,0090 | 0,9752 – 1,000 |
| 3 | 1,000 | 0,9880 ± 0,0100 | 0,9687 – 0,9990 |
| 4 | 0,9989 ± 0,0017 | 0,9710 ± 0,0229 | 0,9168 – 0,9923 |

### Jaccard, k = 3 ([estabilidad_jaccard.csv](src/ejecucion_3/estabilidad_jaccard.csv))

| Clúster (n) | Semillas: media / mín. | Submuestras: media / mín. |
|---|---|---|
| 0 (2461) | 1,000 / 1,000 | 0,9958 / 0,9899 |
| 1 (1023) | 1,000 / 1,000 | 0,9852 / 0,9578 |
| 2 (273) | 1,000 / 1,000 | 0,9822 / 0,9267 |

El detalle de cada repetición está en [estabilidad_detalle.csv](src/ejecucion_3/estabilidad_detalle.csv).

## 6. Consideraciones para interpretar los valores

- **Semillas:** cada ajuste ya se queda con la mejor de 20 inicializaciones, así que es esperable que distintas semillas converjan a la misma solución. Un ARI = 1 indica que el resultado no depende de la inicialización; no dice nada sobre la estabilidad frente a los datos.
- **Submuestras:** los valores altos son en parte esperables por el propio diseño:
  - El modelo de referencia se entrenó con el 100 % de los clientes, que incluye el 80 % usado en cada submuestra.
  - La comparación se hace sobre todos los clientes, incluidos los que se usaron para entrenar. Solo alrededor del 20 % de cada comparación es "fuera de muestra".
- **Preprocesamiento:** el `RobustScaler` y la mediana de imputación se ajustan una sola vez sobre todos los datos, antes del submuestreo. Ese paso no se repite en cada remuestreo.
- **Qué demuestran:** un ARI y un Jaccard altos muestran **robustez frente a perturbaciones moderadas de la muestra**. No constituyen una validación con datos independientes.

## Resumen para la redacción de la tesis

La estabilidad se evaluó mediante **submuestreo aleatorio sin reemplazo** (*subsampling*), no mediante bootstrap ni split-sample. Se generaron 20 submuestras del 80 % de los clientes (3005 de 3757). En cada una se reentrenó K-Means (k-means++, `n_init=20`, `random_state=42`) y el modelo resultante asignó a los 3757 clientes a su centroide más cercano. Esa partición se comparó con la de referencia, entrenada con todos los clientes, mediante el ARI (`adjusted_rand_score`) y un índice de Jaccard por clúster con emparejamiento por máximo solapamiento, sin umbrales fijados en el código. Además, para medir la sensibilidad a la inicialización, se reentrenó el modelo con 10 semillas distintas (1 a 10) sobre la muestra completa.
