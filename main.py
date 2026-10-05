"""
Segmentación de clientes con K-Means (método del codo + silhouette).

Versión en script del notebook elbow-kmeans-silluethe.ipynb.

Uso:
    python main.py --salida src/ejecucion_2

Las credenciales de MySQL se leen de variables de entorno o de un archivo .env
(ver .env.example).
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Los gráficos se guardan en archivos, no se muestran en pantalla

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import RobustScaler
from sqlalchemy import create_engine

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

RANDOM_STATE = 42
N_INIT = 20
K_VALUES = list(range(1, 11))

# silhouette_score es O(n²) en memoria y tiempo; por encima de este número de
# clientes se calcula sobre una muestra aleatoria.
SILHOUETTE_SAMPLE_SIZE = 10000

# Variables numéricas usadas en el clustering
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

# Variables muy sesgadas a las que se aplica log1p
variables_log = [
    "recencia",
    "frecuencia_compra",
    "monto_total",
    "variedad_productos",
    "cantidad_categorias_distintas",
    "cantidad_marcas_distintas"
]

# Porcentajes que quedan NULL si el cliente no tiene productos con categoría/marca
variables_porcentaje = [
    "porcentaje_monto_categoria_top",
    "porcentaje_monto_marca_top"
]

# Variables categóricas: no entran en el clustering, se usan para describir los clústeres
variables_descriptivas = {
    "canal": "canal_predominante",
    "localidad": "localidad_predominante",
    "categoria_predominante": "categoria_predominante_cluster",
    "marca_predominante": "marca_predominante_cluster"
}


# =============================================================================
# CONEXIÓN A MYSQL
# =============================================================================
def crear_engine():
    load_dotenv()

    usuario = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    if not usuario or not password:
        raise SystemExit(
            "Faltan las credenciales de MySQL. Define DB_USER y DB_PASSWORD "
            "como variables de entorno o en un archivo .env (ver .env.example)."
        )

    host = os.getenv("DB_HOST", "localhost")
    puerto = os.getenv("DB_PORT", "3306")
    base = os.getenv("DB_NAME", "renacertfg")

    engine = create_engine(f"mysql+pymysql://{usuario}:{password}@{host}:{puerto}/{base}")

    print("Conexión creada correctamente")
    return engine


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

-- Localidad y canal predominantes: los de mayor monto de compra del cliente
localidad_top AS (
    SELECT
        cliente_id,
        localidad_id,
        ROW_NUMBER() OVER (
            PARTITION BY cliente_id
            ORDER BY SUM(neto_con_imp) DESC, localidad_id
        ) AS rn
    FROM base
    GROUP BY cliente_id, localidad_id
),

canal_top AS (
    SELECT
        cliente_id,
        canal_id,
        ROW_NUMBER() OVER (
            PARTITION BY cliente_id
            ORDER BY SUM(neto_con_imp) DESC, canal_id
        ) AS rn
    FROM base
    GROUP BY cliente_id, canal_id
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
    lt.localidad_id,
    dl.name AS localidad,
    cnt.canal_id,
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
LEFT JOIN localidad_top lt
    ON r.cliente_id = lt.cliente_id AND lt.rn = 1
LEFT JOIN dim_localidades dl
    ON lt.localidad_id = dl.id
LEFT JOIN canal_top cnt
    ON r.cliente_id = cnt.cliente_id AND cnt.rn = 1
LEFT JOIN dim_canales dc
    ON cnt.canal_id = dc.id
LEFT JOIN categoria_top ct
    ON r.cliente_id = ct.cliente_id AND ct.rn = 1
LEFT JOIN marca_top mt
    ON r.cliente_id = mt.cliente_id AND mt.rn = 1
"""


def cargar_datos(engine):
    df = pd.read_sql(query, engine)

    print("Dimensión original:", df.shape)
    print(df.head())
    print(df.isna().sum())

    df["canal"] = df["canal"].fillna("SIN_CANAL")
    df["localidad"] = df["localidad"].fillna("SIN_LOCALIDAD")

    return df


# =============================================================================
# PREPARACIÓN DE LA MATRIZ X_final
# =============================================================================
def preparar_matriz(df):
    X_num = df[variables_cluster].copy()

    # Imputación de porcentajes nulos (clientes sin productos con categoría/marca)
    for col in variables_porcentaje:
        n_nulos = X_num[col].isna().sum()
        if n_nulos > 0:
            mediana = X_num[col].median()
            X_num[col] = X_num[col].fillna(mediana)
            print(f"{col}: {n_nulos} valores nulos imputados con la mediana ({mediana:.2f})")

    for col in variables_log:
        X_num[col] = np.log1p(X_num[col])

    scaler = RobustScaler()

    X_final = pd.DataFrame(
        scaler.fit_transform(X_num),
        columns=X_num.columns,
        index=X_num.index
    )

    if X_final.isna().any().any():
        raise ValueError("X_final contiene valores nulos; revisar la imputación")

    print("Dimensión final de X:", X_final.shape)
    print(X_final.head())

    return X_final


# =============================================================================
# SELECCIÓN DE k: MÉTODO DEL CODO Y SILHOUETTE
# =============================================================================
def evaluar_k(X_final):
    sample_size = SILHOUETTE_SAMPLE_SIZE if len(X_final) > SILHOUETTE_SAMPLE_SIZE else None

    modelos = {}
    filas = []

    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = model.fit_predict(X_final)
        modelos[k] = model

        score = np.nan
        if k >= 2:
            score = silhouette_score(X_final, labels, sample_size=sample_size, random_state=RANDOM_STATE)
            print(f"k={k} -> inercia = {model.inertia_:.2f}, silhouette score = {score:.6f}")

        filas.append({"k": k, "inercia": model.inertia_, "silhouette": score})

    resultados = pd.DataFrame(filas)
    print(resultados)

    kneedle = KneeLocator(
        resultados["k"],
        resultados["inercia"],
        curve="convex",
        direction="decreasing"
    )
    k_optimo = kneedle.elbow

    mejor = resultados.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False).iloc[0]
    best_k = int(mejor["k"])
    best_score = mejor["silhouette"]

    print(f"\nk óptimo según el codo: {k_optimo}")
    print(f"Mejor k según silhouette: {best_k} (score = {best_score:.6f})")
    if k_optimo is not None and k_optimo != best_k:
        print(f"Los métodos no coinciden: el modelo final usa k = {best_k} (silhouette); "
              f"el codo (k = {k_optimo}) se usa solo como referencia.")

    return resultados, k_optimo, best_k, modelos


def graficar_codo(resultados, k_optimo, ruta):
    k_values = resultados["k"].tolist()
    inertias = resultados["inercia"].tolist()

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
    plt.title("Método del codo - variables RFM, variedad y concentración de compra")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()


def graficar_silhouette(resultados, best_k, ruta, k_final=None):
    datos = resultados.dropna(subset=["silhouette"])

    plt.figure(figsize=(8, 5))
    plt.plot(datos["k"], datos["silhouette"], marker='o')
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Mejor k = {best_k}')
    if k_final is not None and k_final != best_k:
        plt.axvline(x=k_final, color='green', linestyle='--', label=f'k elegido = {k_final}')
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Método de Silhouette para validar el número óptimo de clusters")
    plt.xticks(datos["k"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# PERFILADO DE CLÚSTERES
# =============================================================================
def moda(serie):
    return serie.value_counts().index[0] if serie.notna().any() else "SIN_DATO"


def perfilar_clusters(df):
    agrupado = df.groupby("cluster")

    # Se usan medianas: monto_total y frecuencia_compra están muy sesgadas
    resumen_clusters = pd.concat(
        [
            agrupado.size().rename("cantidad_clientes"),
            (agrupado.size() / len(df) * 100).round(2).rename("porcentaje_clientes"),
            agrupado[variables_cluster].median().round(2),
            agrupado[list(variables_descriptivas)].agg(moda).rename(columns=variables_descriptivas)
        ],
        axis=1
    )

    print("\nResumen por clúster (medianas y valores predominantes):")
    print(resumen_clusters)

    # Detalle con media, mediana, mínimo y máximo
    tabla_clusters = agrupado[variables_cluster].agg(["mean", "median", "min", "max"]).round(2)
    print(tabla_clusters)

    return resumen_clusters, tabla_clusters


def graficar_pca(X_final, df, kmeans, ruta):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_final)

    varianza = pca.explained_variance_ratio_
    print(f"\nVarianza explicada por PCA: PC1 = {varianza[0]:.2%}, PC2 = {varianza[1]:.2%}, "
          f"total = {varianza.sum():.2%}")

    # Centroides con nombres de columnas correctos
    centroides_df = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=X_final.columns
    )
    centroides_pca = pca.transform(centroides_df)

    plt.figure(figsize=(11, 7))

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=df["cluster"],
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

    # Etiqueta junto a cada centroide; el índice coincide con el número de clúster
    for cluster, (x, y) in enumerate(centroides_pca):
        plt.annotate(
            f"Cluster {cluster}",
            xy=(x, y),
            xytext=(14, 10),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.85)
        )

    plt.title(f"Segmentación de clientes mediante K-Means "
              f"(PCA: {varianza.sum():.1%} de la varianza explicada)")
    plt.xlabel(f"Componente principal 1 ({varianza[0]:.1%})")
    plt.ylabel(f"Componente principal 2 ({varianza[1]:.1%})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()

    return varianza


# =============================================================================
# VALIDACIÓN: SILUETA POR CLÚSTER Y ESTABILIDAD
# =============================================================================
def silueta_por_cluster(X_final, labels):
    valores = pd.DataFrame({"cluster": labels, "silueta": silhouette_samples(X_final, labels)})

    tabla = valores.groupby("cluster")["silueta"].agg(
        cantidad_clientes="size",
        silueta_media="mean",
        silueta_mediana="median",
        porcentaje_negativos=lambda s: (s < 0).mean() * 100
    ).round(4)

    print("\nSilueta por clúster:")
    print(tabla)
    return tabla


def jaccard_por_cluster(labels_base, labels_nuevas):
    # Para cada clúster original, el mayor Jaccard contra algún clúster de la nueva partición
    resultado = {}
    for c in np.unique(labels_base):
        miembros = labels_base == c
        resultado[c] = max(
            (miembros & (labels_nuevas == j)).sum() / (miembros | (labels_nuevas == j)).sum()
            for j in np.unique(labels_nuevas)
        )
    return resultado


def evaluar_estabilidad(X_final, k, n_semillas=10, n_submuestras=20, fraccion=0.8):
    base = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit_predict(X_final)
    filas = []

    # 1. Sensibilidad a la inicialización: otras semillas, todos los clientes
    for semilla in range(1, n_semillas + 1):
        labels = KMeans(n_clusters=k, random_state=semilla, n_init=N_INIT).fit_predict(X_final)
        filas.append({"k": k, "prueba": "semilla", "repeticion": semilla,
                      "ari": adjusted_rand_score(base, labels),
                      **{f"jaccard_cluster_{c}": v for c, v in jaccard_por_cluster(base, labels).items()}})

    # 2. Sensibilidad a los datos: se entrena con una submuestra y se asigna a todos los clientes
    rng = np.random.default_rng(RANDOM_STATE)
    tamano = int(fraccion * len(X_final))
    for i in range(1, n_submuestras + 1):
        idx = rng.choice(len(X_final), size=tamano, replace=False)
        modelo = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(X_final.iloc[idx])
        labels = modelo.predict(X_final)
        filas.append({"k": k, "prueba": "submuestra", "repeticion": i,
                      "ari": adjusted_rand_score(base, labels),
                      **{f"jaccard_cluster_{c}": v for c, v in jaccard_por_cluster(base, labels).items()}})

    return pd.DataFrame(filas)


def validar_estabilidad(X_final, k_final, k_comparar=(2, 3, 4)):
    detalle = pd.concat(
        [evaluar_estabilidad(X_final, k) for k in sorted(set(k_comparar) | {k_final})],
        ignore_index=True
    )

    resumen = detalle.groupby(["k", "prueba"])["ari"].agg(
        ari_media="mean", ari_desv="std", ari_min="min", ari_max="max", repeticiones="size"
    ).round(4)

    columnas_jaccard = [c for c in detalle.columns if c.startswith("jaccard_cluster_")]
    jaccard = (
        detalle[detalle["k"] == k_final]
        .groupby("prueba")[columnas_jaccard]
        .agg(["mean", "min"])
        .dropna(axis=1, how="all")
        .round(4)
    )

    print("\nEstabilidad (ARI respecto al modelo con random_state=42):")
    print(resumen)
    print(f"\nEstabilidad por clúster con k = {k_final} (índice de Jaccard):")
    print(jaccard)

    return detalle.round(4), resumen, jaccard


COLORES_PRUEBA = {"semilla": "#2a78d6", "submuestra": "#eb6834"}
NOMBRES_PRUEBA = {"semilla": "Otras semillas (10, todos los clientes)",
                  "submuestra": "Submuestras del 80 % (20)"}


def graficar_puntos_estabilidad(ax, grupos, etiquetas):
    # grupos: lista (una entrada por posición en x) de dicts {prueba: valores}
    desplazamiento = {"semilla": -0.18, "submuestra": 0.18}
    rng = np.random.default_rng(RANDOM_STATE)
    for prueba, color in COLORES_PRUEBA.items():
        for pos, grupo in enumerate(grupos):
            valores = np.asarray(grupo[prueba])
            x = pos + desplazamiento[prueba] + rng.uniform(-0.06, 0.06, len(valores))
            ax.scatter(x, valores, s=36, color=color, alpha=0.55, edgecolors="white", linewidths=0.8,
                       label=NOMBRES_PRUEBA[prueba] if pos == 0 else None, zorder=3)
            media = valores.mean()
            ax.hlines(media, pos + desplazamiento[prueba] - 0.13, pos + desplazamiento[prueba] + 0.13,
                      color="#0b0b0b", linewidth=2, zorder=4)
            ax.annotate(f"{media:.3f}", xy=(pos + desplazamiento[prueba] + 0.13, media),
                        xytext=(3, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=9, fontweight="bold", color="#0b0b0b")
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def graficar_estabilidad(detalle, k_final, ruta_ari, ruta_jaccard):
    # ARI por k: cada punto es una repetición, la línea negra es la media
    valores_k = sorted(detalle["k"].unique())
    grupos = [{p: detalle[(detalle["k"] == k) & (detalle["prueba"] == p)]["ari"] for p in COLORES_PRUEBA}
              for k in valores_k]
    etiquetas = [f"k = {k}" + (" (final)" if k == k_final else "") for k in valores_k]

    fig, ax = plt.subplots(figsize=(10, 6))
    graficar_puntos_estabilidad(ax, grupos, etiquetas)
    ax.set_ylim(min(0.9, detalle["ari"].min() - 0.01), 1.005)
    ax.set_title("Estabilidad de K-Means: ARI respecto al modelo de referencia (random_state=42)")
    ax.set_xlabel("Número de clústeres")
    ax.set_ylabel("Índice Rand Ajustado (ARI)\n(eje recortado: el ARI vale 1 como máximo y puede ser negativo)")
    ax.legend(title="Línea negra y valor: media", loc="lower left", frameon=False)
    fig.savefig(ruta_ari, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Jaccard por clúster para el k final
    final = detalle[detalle["k"] == k_final]
    clusters = range(k_final)
    grupos = [{p: final[final["prueba"] == p][f"jaccard_cluster_{c}"] for p in COLORES_PRUEBA}
              for c in clusters]

    fig, ax = plt.subplots(figsize=(10, 6))
    graficar_puntos_estabilidad(ax, grupos, [f"Cluster {c}" for c in clusters])
    minimo = min(final[f"jaccard_cluster_{c}"].min() for c in clusters)
    ax.set_ylim(min(0.9, minimo - 0.01), 1.005)
    ax.set_title(f"Estabilidad por clúster (k = {k_final}): índice de Jaccard máximo frente a cada repetición")
    ax.set_xlabel("Clúster del modelo de referencia")
    ax.set_ylabel("Índice de Jaccard\n(eje recortado: la escala completa va de 0 a 1)")
    ax.legend(title="Línea negra y valor: media", loc="lower left", frameon=False)
    fig.savefig(ruta_jaccard, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# EXPORTAR MUESTRA DE X_final COMO IMAGEN
# =============================================================================
def exportar_muestra_xfinal(X_final, ruta):
    # Seleccionar una muestra pequeña de X_final y redondear para que se vea mejor
    tabla_xfinal = X_final.head(8).iloc[:, :10].round(3)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    tabla = ax.table(
        cellText=tabla_xfinal.values,
        # Nombres partidos en varias líneas para que no se superpongan
        colLabels=[col.replace("_", "\n") for col in tabla_xfinal.columns],
        rowLabels=tabla_xfinal.index,
        cellLoc="center",
        loc="center"
    )

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8)
    tabla.scale(1, 1.4)
    for (fila, _), celda in tabla.get_celld().items():
        if fila == 0:
            celda.set_height(celda.get_height() * 3)

    plt.title("Vista parcial de la matriz analítica final X_final", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Imagen guardada en: {ruta}")


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Segmentación de clientes con K-Means")
    parser.add_argument("--salida", default="resultados",
                        help="Carpeta donde se guardan gráficos y tablas (por defecto: resultados)")
    parser.add_argument("--k", type=int, choices=K_VALUES[1:],
                        help="Fija el número de clústeres del modelo final "
                             "(por defecto: el de mejor silhouette)")
    args = parser.parse_args()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    engine = crear_engine()
    df = cargar_datos(engine)
    X_final = preparar_matriz(df)

    resultados, k_optimo, best_k, modelos = evaluar_k(X_final)
    graficar_codo(resultados, k_optimo, salida / "01_codo.png")
    graficar_silhouette(resultados, best_k, salida / "02_silhouette.png", args.k)

    # Modelo final: el de mejor silhouette (o el k fijado con --k), ya entrenado en evaluar_k
    k_final = best_k
    if args.k is not None:
        k_final = args.k
        score = resultados.loc[resultados["k"] == k_final, "silhouette"].iloc[0]
        print(f"\nk fijado manualmente: {k_final} (silhouette = {score:.6f}); "
              f"el mejor según silhouette es k = {best_k}")

    kmeans = modelos[k_final]
    df["cluster"] = kmeans.labels_

    resumen_clusters, tabla_clusters = perfilar_clusters(df)
    graficar_pca(X_final, df, kmeans, salida / "03_pca_clusters.png")
    exportar_muestra_xfinal(X_final, salida / "04_xfinal_muestra.png")

    silueta_clusters = silueta_por_cluster(X_final, kmeans.labels_)
    estabilidad_detalle, estabilidad_resumen, estabilidad_jaccard = validar_estabilidad(X_final, k_final)
    graficar_estabilidad(estabilidad_detalle, k_final,
                         salida / "05_estabilidad_ari.png", salida / "06_estabilidad_jaccard.png")

    silueta_clusters.to_csv(salida / "silueta_por_cluster.csv")
    estabilidad_detalle.to_csv(salida / "estabilidad_detalle.csv", index=False)
    estabilidad_resumen.to_csv(salida / "estabilidad_resumen.csv")
    estabilidad_jaccard.to_csv(salida / "estabilidad_jaccard.csv")
    resultados.to_csv(salida / "metricas_k.csv", index=False)
    resumen_clusters.to_csv(salida / "resumen_clusters.csv")
    tabla_clusters.to_csv(salida / "detalle_clusters.csv")
    df[["cliente_id", "cluster"]].to_csv(salida / "clientes_cluster.csv", index=False)

    print(f"\nResultados guardados en: {salida.resolve()}")


if __name__ == "__main__":
    main()
