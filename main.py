"""
Segmentación de clientes con K-Means (método del codo + silhouette).

Versión en script del notebook elbow-kmeans-silluethe.ipynb.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
from sqlalchemy import create_engine

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# =============================================================================
# CONEXIÓN A MYSQL
# =============================================================================
# Valores por defecto iguales a los del notebook; se pueden sobrescribir con
# variables de entorno.
usuario = os.getenv("DB_USER", "root")
password = os.getenv("DB_PASSWORD", "root")
host = os.getenv("DB_HOST", "localhost")
puerto = os.getenv("DB_PORT", "3306")
base = os.getenv("DB_NAME", "renacertfg")

engine = create_engine(f"mysql+pymysql://{usuario}:{password}@{host}:{puerto}/{base}")

print("Conexión creada correctamente")


# =============================================================================
# EXTRACCIÓN DE DATOS POR CLIENTE
# =============================================================================
query = """
WITH base AS (
    SELECT
        fv.cliente_id,
        fv.localidad_id,
        fv.canal_id,
        fv.fecha,
        fv.documento_id,
        fv.serie,
        fv.numero,
        fv.producto_code,
        COALESCE(fv.neto_con_imp, 0) AS neto_con_imp,

        dp.categoria_id,
        cat.name AS categoria,
        dp.marca_id,
        mar.name AS marca

    FROM vw_fact_ventas_tipada fv
    LEFT JOIN dim_productos dp
        ON fv.producto_code = dp.code
    LEFT JOIN dim_categorias cat
        ON dp.categoria_id = cat.id
    LEFT JOIN dim_marcas mar
        ON dp.marca_id = mar.id

    WHERE fv.fecha BETWEEN '2022-01-01' AND '2024-12-31'
      AND fv.cliente_id IS NOT NULL
      AND COALESCE(fv.neto_con_imp, 0) > 0
),

rfm AS (
    SELECT
        cliente_id,
        MAX(localidad_id) AS localidad_id,
        MAX(canal_id) AS canal_id,
        DATEDIFF('2024-12-31', MAX(fecha)) AS recencia,
        COUNT(DISTINCT CONCAT(
            COALESCE(documento_id, ''), '-',
            COALESCE(serie, ''), '-',
            COALESCE(numero, '')
        )) AS frecuencia_compra,
        SUM(neto_con_imp) AS monto_total,
        COUNT(DISTINCT producto_code) AS variedad_productos,
        COUNT(DISTINCT categoria_id) AS cantidad_categorias_distintas,
        COUNT(DISTINCT marca_id) AS cantidad_marcas_distintas
    FROM base
    GROUP BY cliente_id
),

categoria_monto AS (
    SELECT
        cliente_id,
        categoria_id,
        categoria,
        SUM(neto_con_imp) AS monto_categoria
    FROM base
    WHERE categoria_id IS NOT NULL
    GROUP BY cliente_id, categoria_id, categoria
),

categoria_top AS (
    SELECT
        cliente_id,
        categoria_id,
        categoria,
        monto_categoria,
        ROW_NUMBER() OVER (
            PARTITION BY cliente_id
            ORDER BY monto_categoria DESC, categoria_id
        ) AS rn
    FROM categoria_monto
),

marca_monto AS (
    SELECT
        cliente_id,
        marca_id,
        marca,
        SUM(neto_con_imp) AS monto_marca
    FROM base
    WHERE marca_id IS NOT NULL
    GROUP BY cliente_id, marca_id, marca
),

marca_top AS (
    SELECT
        cliente_id,
        marca_id,
        marca,
        monto_marca,
        ROW_NUMBER() OVER (
            PARTITION BY cliente_id
            ORDER BY monto_marca DESC, marca_id
        ) AS rn
    FROM marca_monto
)

SELECT
    r.cliente_id,
    r.localidad_id,
    dl.name AS localidad,
    r.canal_id,
    dc.name AS canal,

    r.recencia,
    r.frecuencia_compra,
    r.monto_total,
    r.variedad_productos,

    r.cantidad_categorias_distintas,
    r.cantidad_marcas_distintas,

    ct.categoria AS categoria_predominante,
    ROUND((ct.monto_categoria / NULLIF(r.monto_total, 0)) * 100, 2) AS porcentaje_monto_categoria_top,

    mt.marca AS marca_predominante,
    ROUND((mt.monto_marca / NULLIF(r.monto_total, 0)) * 100, 2) AS porcentaje_monto_marca_top

FROM rfm r
LEFT JOIN dim_localidades dl
    ON r.localidad_id = dl.id
LEFT JOIN dim_canales dc
    ON r.canal_id = dc.id
LEFT JOIN categoria_top ct
    ON r.cliente_id = ct.cliente_id AND ct.rn = 1
LEFT JOIN marca_top mt
    ON r.cliente_id = mt.cliente_id AND mt.rn = 1
"""

df = pd.read_sql(query, engine)

print("Dimensión original:", df.shape)
print(df.head())
print(df.isna().sum())


# =============================================================================
# PREPARACIÓN DE LA MATRIZ X_final
# =============================================================================
variables_cluster = [
    "recencia",
    "frecuencia_compra",
    "monto_total",
    "variedad_productos",
    "cantidad_categorias_distintas",
    "cantidad_marcas_distintas",
    "porcentaje_monto_categoria_top",
    "porcentaje_monto_marca_top"
]

X_num = df[variables_cluster].copy()
print(X_num.head())

X_num["recencia"] = np.log1p(X_num["recencia"])
X_num["frecuencia_compra"] = np.log1p(X_num["frecuencia_compra"])
X_num["monto_total"] = np.log1p(X_num["monto_total"])
X_num["variedad_productos"] = np.log1p(X_num["variedad_productos"])
X_num["cantidad_categorias_distintas"] = np.log1p(X_num["cantidad_categorias_distintas"])
X_num["cantidad_marcas_distintas"] = np.log1p(X_num["cantidad_marcas_distintas"])

print(X_num.head())

X_cat = df[["canal", "localidad"]].copy()

X_cat["canal"] = X_cat["canal"].fillna("SIN_CANAL")
X_cat["localidad"] = X_cat["localidad"].fillna("SIN_LOCALIDAD")

print(X_cat.head())

X_cat_dummies = pd.get_dummies(
    X_cat,
    columns=["canal", "localidad"],
    drop_first=False,
    dtype=int
)

print("Dimensión de variables categóricas dummy:", X_cat_dummies.shape)
print(X_cat_dummies.head())

scaler = RobustScaler()

X_num_scaled = pd.DataFrame(
    scaler.fit_transform(X_num),
    columns=X_num.columns,
    index=X_num.index
)

print(X_num_scaled.head())

X_final = pd.concat([X_num_scaled, X_cat_dummies], axis=1)

print("Dimensión final de X:", X_final.shape)
print(X_final.head())


# =============================================================================
# MÉTODO DEL CODO - ELBOW
# =============================================================================
k_values = list(range(1, 11))
inertias = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    model.fit(X_final)
    inertias.append(model.inertia_)

resultados = pd.DataFrame({
    "k": k_values,
    "inercia": inertias
})

print(resultados)

kneedle = KneeLocator(
    k_values,
    inertias,
    curve="convex",
    direction="decreasing"
)

k_optimo = kneedle.elbow
print("k óptimo detectado:", k_optimo)

plt.figure(figsize=(10, 6))
plt.plot(k_values, inertias, marker='o', linewidth=2, label='Inercia')

if k_optimo is not None:
    y_optimo = inertias[k_values.index(k_optimo)]
    plt.axvline(x=k_optimo, color='red', linestyle='--', label=f'k óptimo = {k_optimo}')
    plt.scatter(k_optimo, y_optimo, s=80, color='red')
    plt.annotate(
        f'k = {k_optimo}',
        xy=(k_optimo, y_optimo),
        xytext=(k_optimo + 0.3, y_optimo + 0.05 * max(inertias)),
        arrowprops=dict(arrowstyle='->'),
        fontsize=11
    )

plt.xticks(k_values)
plt.xlabel("Número de clústeres (k)")
plt.ylabel("Inercia")
plt.title("Método del codo - RFM + canal + localidad + categorías + marcas")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# =============================================================================
# SILHOUETTE
# =============================================================================
# Lista para guardar los resultados
silhouette_scores = []

# Rango de valores de k a evaluar
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_final)
    score = silhouette_score(X_final, labels)
    silhouette_scores.append(score)
    print(f"k={k} -> silhouette score = {score:.6f}")

# Gráfico del método de silhouette
plt.figure(figsize=(8, 5))
plt.plot(K, silhouette_scores, marker='o')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Método de Silhouette para validar el número óptimo de clusters")
plt.xticks(K)
plt.grid(True, alpha=0.3)
plt.show()

# Mejor k según silhouette
best_k = K[silhouette_scores.index(max(silhouette_scores))]
best_score = max(silhouette_scores)

print(f"\nMejor k según silhouette: {best_k}")
print(f"Mejor silhouette score: {best_score:.6f}")


# =============================================================================
# MODELO K-MEANS
# =============================================================================
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df["cluster"] = kmeans.fit_predict(X_final)

print(df.head())

conteo_clusters = df["cluster"].value_counts().sort_index()
print(conteo_clusters)

perfil_clusters = df.groupby("cluster")[variables_cluster].mean().round(2)

print(perfil_clusters)

canal_por_cluster = (
    df.groupby("cluster")["canal"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
)
canal_por_cluster.columns = ["cluster", "canal_predominante"]

localidad_por_cluster = (
    df.groupby("cluster")["localidad"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
)
localidad_por_cluster.columns = ["cluster", "localidad_predominante"]

print(canal_por_cluster)
print(localidad_por_cluster)

categoria_por_cluster = (
    df.groupby("cluster")["categoria_predominante"]
    .agg(lambda x: x.value_counts().index[0] if x.notna().any() else "SIN_DATO")
    .reset_index()
)
categoria_por_cluster.columns = ["cluster", "categoria_predominante_cluster"]

marca_por_cluster = (
    df.groupby("cluster")["marca_predominante"]
    .agg(lambda x: x.value_counts().index[0] if x.notna().any() else "SIN_DATO")
    .reset_index()
)
marca_por_cluster.columns = ["cluster", "marca_predominante_cluster"]

print(categoria_por_cluster)
print(marca_por_cluster)

conteo_clusters_df = df["cluster"].value_counts().sort_index().reset_index()
conteo_clusters_df.columns = ["cluster", "cantidad_clientes"]

porcentaje_clusters = (
    df["cluster"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
    .reset_index()
)
porcentaje_clusters.columns = ["cluster", "porcentaje_clientes"]

resumen_clusters = (
    perfil_clusters.reset_index()
    .merge(conteo_clusters_df, on="cluster")
    .merge(porcentaje_clusters, on="cluster")
    .merge(canal_por_cluster, on="cluster")
    .merge(localidad_por_cluster, on="cluster")
    .merge(categoria_por_cluster, on="cluster")
    .merge(marca_por_cluster, on="cluster")
)

print(resumen_clusters)

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_final)

# Datos para gráfico
df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=df.index)
df_plot["cluster"] = df["cluster"]

# Centroides con nombres de columnas correctos
centroides_df = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=X_final.columns
)
centroides_pca = pca.transform(centroides_df)

# Visualización
plt.figure(figsize=(11, 7))

scatter = plt.scatter(
    df_plot["PC1"],
    df_plot["PC2"],
    c=df_plot["cluster"],
    cmap="viridis",
    alpha=0.7
)

plt.scatter(
    centroides_pca[:, 0],
    centroides_pca[:, 1],
    s=350,
    marker="X",
    c="red",
    edgecolors="black",
    label="Centroides"
)

plt.title("Segmentación de clientes mediante K-Means")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Tabla resumen por cluster
tabla_clusters = df.groupby("cluster")[variables_cluster].agg([
    "mean",
    "median",
    "min",
    "max"
]).round(2)

print(tabla_clusters)

# Cantidad de clientes por cluster
cantidad_clientes = df.groupby("cluster").size().rename("cantidad_clientes")

# Promedio de variables por cluster
promedios_clusters = df.groupby("cluster")[variables_cluster].mean().round(2)

# Unir en una sola tabla
tabla_resumen = pd.concat([cantidad_clientes, promedios_clusters], axis=1)

print(tabla_resumen)


# =============================================================================
# EXPORTAR MUESTRA DE X_final COMO IMAGEN
# =============================================================================
# Seleccionar una muestra pequeña de X_final
tabla_xfinal = X_final.head(8).iloc[:, :10].copy()

# Redondear valores para que se vea mejor
tabla_xfinal = tabla_xfinal.round(3)

# Crear figura
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

tabla = ax.table(
    cellText=tabla_xfinal.values,
    colLabels=tabla_xfinal.columns,
    rowLabels=tabla_xfinal.index,
    cellLoc="center",
    loc="center"
)

tabla.auto_set_font_size(False)
tabla.set_fontsize(8)
tabla.scale(1, 1.4)

plt.title("Vista parcial de la matriz analítica final X_final", fontsize=12, pad=15)
plt.tight_layout()

# Guardar en la carpeta Descargas
ruta_descargas = Path.home() / "Downloads" / "xfinal_muestra.png"

plt.savefig(ruta_descargas, dpi=300, bbox_inches="tight")
plt.show()

print(f"Imagen guardada en: {ruta_descargas}")
