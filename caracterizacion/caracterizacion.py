"""
Caracterización gráfica de los clústeres de K-Means y detección de clientes en el
límite ("limbo") entre dos clústeres.

Reutiliza la extracción y la preparación de datos de main.py y vuelve a entrenar
KMeans con los mismos parámetros (RANDOM_STATE, N_INIT), por lo que las etiquetas
coinciden con las de clientes_cluster.csv de una ejecución de main.py.

Uso (desde la raíz del proyecto):
    python caracterizacion/caracterizacion.py
    python caracterizacion/caracterizacion.py --k 2 3 --limbo 0-1 1-2 --datos datos.pkl

Estructura de salida (por defecto caracterizacion/resultados/):
    graficos_y_reportes/k<k>/        gráficos y tablas de caracterización de cada k
    limbo/cluster_<a>_cluster_<b>/   clientes en el límite entre dos clústeres (k = --k-limbo)
"""

import argparse
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# main.py está en la carpeta superior
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from main import (N_INIT, RANDOM_STATE, cargar_datos, crear_engine, graficar_silueta_clientes,
                  preparar_matriz, variables_cluster, variables_log)

# Paleta categórica validada (azul, naranja, aqua): los tres colores se distinguen
# entre sí también con daltonismo. Los clústeres se rotulan siempre con texto.
COLORES_CLUSTER = ["#2a78d6", "#eb6834", "#1baf7a"]
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRIS = "#b5b3ad"
SUPERFICIE = "#fcfcfb"

# Secuencial (un solo tono, de claro a oscuro) y divergente (azul - gris - rojo)
CMAP_SECUENCIAL = LinearSegmentedColormap.from_list(
    "azul", ["#f4f8fd", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
CMAP_DIVERGENTE = LinearSegmentedColormap.from_list(
    "azul_rojo", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a99", "#b8312f"])

NOMBRES = {
    "recencia": "Recencia (días)",
    "frecuencia_compra": "Frecuencia (compras)",
    "monto_total": "Monto total",
    "variedad_productos": "Variedad de productos",
    "cantidad_categorias_distintas": "Categorías distintas",
    "cantidad_marcas_distintas": "Marcas distintas",
    "porcentaje_monto_categoria_top": "% monto categoría top",
    "porcentaje_monto_marca_top": "% monto marca top",
}

NOMBRES_CORTOS = {
    "recencia": "Recencia",
    "frecuencia_compra": "Frecuencia",
    "monto_total": "Monto\ntotal",
    "variedad_productos": "Variedad\nproductos",
    "cantidad_categorias_distintas": "Categorías\ndistintas",
    "cantidad_marcas_distintas": "Marcas\ndistintas",
    "porcentaje_monto_categoria_top": "% categoría\ntop",
    "porcentaje_monto_marca_top": "% marca\ntop",
}

DESCRIPTIVAS = {
    "canal": "Canal predominante",
    "localidad": "Localidad predominante",
    "categoria_predominante": "Categoría predominante",
    "marca_predominante": "Marca predominante",
}

# Limbo: el cliente está a menos de este porcentaje de distancia de quedar a igual
# distancia de los dos centroides (margen relativo = (d_otro - d_propio) / d_otro)
UMBRAL_LIMBO = 0.10
UMBRAL_LIMBO_CRITICO = 0.05
N_SUBMUESTRAS_LIMBO = 50


def estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": GRIS,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": TINTA_SECUNDARIA,
        "ytick.color": TINTA_SECUNDARIA,
        "legend.frameon": False,
        "font.size": 10,
    })


def color(c):
    return COLORES_CLUSTER[c % len(COLORES_CLUSTER)]


def formato(valor, variable):
    if variable.startswith("porcentaje"):
        return f"{valor:.1f} %"
    if abs(valor) >= 1e6:
        return f"{valor / 1e6:.1f} M"
    if abs(valor) >= 1e3:
        return f"{valor / 1e3:.1f} mil"
    return f"{valor:.0f}"


def guardar(fig, ruta):
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {ruta}")


def escala_eje(ax, variable, minimo, eje="y"):
    # Las variables de conteo y monto están muy sesgadas: escala logarítmica
    # (symlog anclada en 0 si hay ceros, por ejemplo recencia = 0)
    if variable not in variables_log:
        return
    eje_mpl = ax.xaxis if eje == "x" else ax.yaxis
    if minimo > 0:
        getattr(ax, f"set_{eje}scale")("log")
    else:
        getattr(ax, f"set_{eje}scale")("symlog", linthresh=1)
        if eje == "x":
            ax.set_xlim(left=0)
        else:
            ax.set_ylim(bottom=0)
    eje_mpl.set_major_locator(LogLocator(base=10, numticks=12))
    eje_mpl.set_major_formatter(FuncFormatter(lambda v, _: numero_legible(v)))
    eje_mpl.set_minor_formatter(NullFormatter())


def numero_legible(v):
    if v >= 1e6:
        return f"{v / 1e6:g} M"
    if v >= 1e3:
        return f"{v / 1e3:g} mil"
    return f"{v:g}"


# =============================================================================
# GRÁFICOS DE CARACTERIZACIÓN
# =============================================================================
def graficar_tamano(df, k, ruta):
    clusters = range(k)
    pct_clientes = df["cluster"].value_counts(normalize=True).sort_index() * 100
    pct_monto = df.groupby("cluster")["monto_total"].sum() / df["monto_total"].sum() * 100
    n = df["cluster"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, valores, titulo in [(axes[0], pct_clientes, "% de clientes"),
                                (axes[1], pct_monto, "% del monto total facturado")]:
        barras = ax.bar([f"Cluster {c}" for c in clusters], valores, width=0.6,
                        color=[color(c) for c in clusters], edgecolor=SUPERFICIE, linewidth=2)
        for c, barra in zip(clusters, barras):
            texto = f"{valores[c]:.1f} %"
            if titulo.startswith("% de clientes"):
                texto += f"\n({n[c]:,} clientes)".replace(",", ".")
            ax.annotate(texto, (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        color=TINTA, fontsize=10)
        ax.set_title(titulo)
        ax.set_ylim(0, 110)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Porcentaje")
    fig.suptitle(f"Tamaño e importancia económica de cada clúster (k = {k})", fontweight="bold")
    guardar(fig, ruta)


def graficar_boxplots(df, k, ruta, grupos=None, etiquetas=None, colores=None, rayados=None, titulo=None):
    # Por defecto, una caja por clúster; se puede pasar otra agrupación (máscaras booleanas)
    if grupos is None:
        grupos = [df["cluster"] == c for c in range(k)]
        etiquetas = [f"C{c}" for c in range(k)]
        colores = [color(c) for c in range(k)]
    rayados = rayados or [None] * len(grupos)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, variable in zip(axes.flat, variables_cluster):
        datos = [df.loc[g, variable].dropna() for g in grupos]
        cajas = ax.boxplot(datos, positions=range(len(grupos)), widths=0.6, patch_artist=True,
                           medianprops=dict(color=TINTA, linewidth=2),
                           flierprops=dict(marker="o", markersize=2.5, markerfacecolor=GRIS,
                                           markeredgecolor="none", alpha=0.6),
                           whiskerprops=dict(color=TINTA_SECUNDARIA),
                           capprops=dict(color=TINTA_SECUNDARIA))
        for caja, col, rayado in zip(cajas["boxes"], colores, rayados):
            caja.set_facecolor(col)
            caja.set_alpha(0.85)
            caja.set_edgecolor(TINTA_SECUNDARIA if rayado is None else SUPERFICIE)
            if rayado:
                caja.set_hatch(rayado)
        ax.set_xticks(range(len(grupos)))
        ax.set_xticklabels(etiquetas, fontsize=9)
        ax.set_title(NOMBRES[variable])
        escala_eje(ax, variable, df[variable].min())
        ax.grid(axis="x", visible=False)
    fig.suptitle(titulo or f"Distribución de las variables de clustering por clúster (k = {k}); "
                 "línea negra = mediana, escala logarítmica en conteos y montos",
                 fontweight="bold")
    fig.tight_layout()
    guardar(fig, ruta)


def percentiles_por_cluster(df, k):
    # Percentil (0-100) que ocupa la mediana de cada clúster dentro de toda la base
    rangos = df[variables_cluster].rank(pct=True) * 100
    rangos["cluster"] = df["cluster"]
    return rangos.groupby("cluster")[variables_cluster].median().reindex(range(k))


def graficar_heatmap_perfil(df, k, ruta):
    pct = percentiles_por_cluster(df, k)
    medianas = df.groupby("cluster")[variables_cluster].median()
    n = df["cluster"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(14, 1.2 * k + 2))
    imagen = ax.imshow(pct.values, cmap=CMAP_DIVERGENTE, norm=TwoSlopeNorm(50, 0, 100), aspect="auto")
    for i in range(k):
        for j, variable in enumerate(variables_cluster):
            p = pct.iloc[i, j]
            ax.text(j, i, f"P{p:.0f}\n{formato(medianas.iloc[i, j], variable)}", ha="center",
                    va="center", fontsize=9.5, color="white" if abs(p - 50) > 30 else TINTA)
    ax.set_xticks(range(len(variables_cluster)))
    ax.set_xticklabels([NOMBRES_CORTOS[v] for v in variables_cluster])
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"Cluster {c}\n(n = {n[c]})" for c in range(k)])
    ax.grid(False)
    ax.tick_params(length=0)
    for borde in ax.spines.values():
        borde.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(variables_cluster)), minor=True)
    ax.set_yticks(np.arange(-0.5, k), minor=True)
    ax.grid(which="minor", color=SUPERFICIE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    barra = fig.colorbar(imagen, ax=ax, fraction=0.03, pad=0.02)
    barra.set_label("Percentil de la mediana del clúster en toda la base")
    barra.set_ticks([0, 25, 50, 75, 100])
    ax.set_title(f"Perfil de los clústeres (k = {k}): percentil y mediana de cada variable\n"
                 "P50 = cliente típico de la base; azul = por debajo, rojo = por encima", loc="left")
    guardar(fig, ruta)
    return pct, medianas


def separar(posiciones, distancia):
    # Desplaza etiquetas (manteniendo su orden) para que queden al menos a `distancia`
    orden = np.argsort(posiciones)
    resultado = np.asarray(posiciones, dtype=float).copy()
    for anterior, actual in zip(orden[:-1], orden[1:]):
        resultado[actual] = max(resultado[actual], resultado[anterior] + distancia)
    # Se recentra para que el desplazamiento quede repartido arriba y abajo
    return resultado - (resultado.mean() - np.mean(posiciones))


def graficar_snake(X_final, labels, k, ruta):
    medias = X_final.groupby(labels).mean()
    x = np.arange(len(variables_cluster))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.axhline(0, color=TINTA_SECUNDARIA, linewidth=1, linestyle="--", label="Cliente mediano de la base")
    for c in range(k):
        ax.plot(x, medias.loc[c], color=color(c), linewidth=2, marker="o", markersize=8,
                markeredgecolor=SUPERFICIE, markeredgewidth=2, label=f"Cluster {c}")
    finales = medias.iloc[:, -1].values
    rango = medias.values.max() - medias.values.min()
    for c, y in zip(range(k), separar(finales, 0.07 * rango)):
        ax.annotate(f"Cluster {c}", (x[-1], finales[c]), xytext=(x[-1] + 0.15, y), va="center",
                    fontsize=10, fontweight="bold", color=TINTA,
                    arrowprops=dict(arrowstyle="-", color=GRIS, linewidth=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels([NOMBRES_CORTOS[v] for v in variables_cluster])
    ax.set_xlim(-0.3, len(x) - 0.2)
    ax.set_ylabel("Media en la escala del modelo\n(log1p + RobustScaler)")
    ax.set_title(f"Snake plot: perfil medio de cada clúster en las variables del modelo (k = {k})", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=k + 1)
    guardar(fig, ruta)


def graficar_radar(df, k, ruta):
    pct = percentiles_por_cluster(df, k)
    angulos = np.linspace(0, 2 * np.pi, len(variables_cluster), endpoint=False).tolist()
    angulos += angulos[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for c in range(k):
        valores = pct.loc[c].tolist()
        valores += valores[:1]
        ax.plot(angulos, valores, color=color(c), linewidth=2, label=f"Cluster {c}")
        ax.fill(angulos, valores, color=color(c), alpha=0.12)
        ax.scatter(angulos[:-1], valores[:-1], s=40, color=color(c), edgecolors=SUPERFICIE,
                   linewidths=1.5, zorder=3)
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels([NOMBRES_CORTOS[v] for v in variables_cluster], color=TINTA)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["P25", "P50", "P75", "P100"], color=TINTA_SECUNDARIA, fontsize=8)
    ax.spines["polar"].set_color(GRIS)
    ax.set_title(f"Radar de perfiles (k = {k}): percentil de la mediana de cada clúster",
                 pad=25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=k)
    guardar(fig, ruta)


def graficar_dispersion(df, k, ruta):
    pares = [("frecuencia_compra", "monto_total"),
             ("recencia", "frecuencia_compra"),
             ("variedad_productos", "monto_total")]
    # Primero el clúster más grande, así los pequeños quedan visibles encima
    orden = df["cluster"].value_counts().index

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, (vx, vy) in zip(axes, pares):
        for c in orden:
            sub = df[df["cluster"] == c]
            ax.scatter(sub[vx], sub[vy], s=12, color=color(c), alpha=0.55, edgecolors=SUPERFICIE,
                       linewidths=0.3, label=f"Cluster {c}")
        escala_eje(ax, vx, df[vx].min(), "x")
        escala_eje(ax, vy, df[vy].min(), "y")
        ax.set_xlabel(NOMBRES[vx])
        ax.set_ylabel(NOMBRES[vy])
    manejadores = [Patch(color=color(c), label=f"Cluster {c}") for c in range(k)]
    fig.legend(handles=manejadores, loc="lower center", ncol=k, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Relación entre variables RFM y variedad por clúster (k = {k}), escala logarítmica",
                 fontweight="bold")
    fig.tight_layout()
    guardar(fig, ruta)


def graficar_composicion(df, k, columna, ruta, top=10):
    # % de clientes de cada clúster según su valor predominante (las menos frecuentes van a "Otros")
    valores = df[columna].fillna("SIN_DATO")
    principales = valores.value_counts().index[:top]
    valores = valores.where(valores.isin(principales), "Otros")
    tabla = pd.crosstab(df["cluster"], valores, normalize="index") * 100
    columnas = list(principales) + (["Otros"] if "Otros" in tabla.columns else [])
    tabla = tabla.reindex(columns=columnas, fill_value=0).reindex(range(k))

    fig, ax = plt.subplots(figsize=(max(10, 1.15 * len(columnas) + 2), 1.1 * k + 2.4))
    imagen = ax.imshow(tabla.values, cmap=CMAP_SECUENCIAL, vmin=0, vmax=max(50, tabla.values.max()),
                       aspect="auto")
    limite = imagen.norm.vmax * 0.55
    for i in range(k):
        for j in range(len(columnas)):
            v = tabla.iloc[i, j]
            ax.text(j, i, f"{v:.1f} %", ha="center", va="center", fontsize=9,
                    color="white" if v > limite else TINTA)
    ax.set_xticks(range(len(columnas)))
    ax.set_xticklabels([textwrap.fill(str(c).strip().replace("/", "/ "), 11) for c in columnas], fontsize=8.5)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"Cluster {c}" for c in range(k)])
    ax.grid(False)
    ax.tick_params(length=0)
    for borde in ax.spines.values():
        borde.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(columnas)), minor=True)
    ax.set_yticks(np.arange(-0.5, k), minor=True)
    ax.grid(which="minor", color=SUPERFICIE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    barra = fig.colorbar(imagen, ax=ax, fraction=0.03, pad=0.02)
    barra.set_label("% de clientes del clúster")
    ax.set_title(f"{DESCRIPTIVAS[columna]} por clúster (k = {k}): % de clientes de cada fila\n"
                 f"(los {top} valores más frecuentes de la base; el resto en \"Otros\")", loc="left")
    guardar(fig, ruta)
    return tabla


def graficar_biplot(X_final, labels, kmeans, k, ruta):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_final)
    varianza = pca.explained_variance_ratio_
    centroides = pca.transform(pd.DataFrame(kmeans.cluster_centers_, columns=X_final.columns))
    # Correlación de cada variable con cada componente (círculo de correlaciones)
    correlaciones = np.array([[np.corrcoef(X_final[v], X_pca[:, i])[0, 1] for i in range(2)]
                              for v in X_final.columns])

    fig, (ax, ax_circ) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw=dict(width_ratios=[1.5, 1]))
    for c in pd.Series(labels).value_counts().index:
        ax.scatter(X_pca[labels == c, 0], X_pca[labels == c, 1], s=10, color=color(c), alpha=0.35,
                   edgecolors="none", label=f"Cluster {c}")
    for c, (x, y) in enumerate(centroides):
        ax.scatter(x, y, s=260, marker="X", color=color(c), edgecolors=TINTA, linewidths=1.5, zorder=4)
        ax.annotate(f"Centroide {c}", (x, y), xytext=(10, -16), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=TINTA,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SUPERFICIE, edgecolor="none", alpha=0.85))
    ax.axhline(0, color=GRIS, linewidth=0.8)
    ax.axvline(0, color=GRIS, linewidth=0.8)
    ax.set_xlabel(f"Componente principal 1 ({varianza[0]:.1%})")
    ax.set_ylabel(f"Componente principal 2 ({varianza[1]:.1%})")
    ax.set_title("Clientes y centroides en el plano PCA", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=k, markerscale=2.5)

    ax_circ.add_patch(plt.Circle((0, 0), 1, fill=False, color=GRIS, linewidth=1))
    for (x, y) in correlaciones:
        ax_circ.annotate("", xy=(x, y), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=TINTA, linewidth=1.5))
    # Etiquetas a la derecha o a la izquierda según el signo, separadas en vertical
    for lado in (1, -1):
        idx = [j for j, (x, _) in enumerate(correlaciones) if np.sign(x) == lado or (x == 0 and lado == 1)]
        if not idx:
            continue
        ys = separar(correlaciones[idx, 1], 0.09)
        for j, y in zip(idx, ys):
            x = correlaciones[j, 0]
            ax_circ.annotate(NOMBRES_CORTOS[X_final.columns[j]].replace("\n", " "), (x, correlaciones[j, 1]),
                             xytext=(x + 0.08 * lado, y), ha="left" if lado == 1 else "right", va="center",
                             fontsize=9.5, color=TINTA,
                             arrowprops=dict(arrowstyle="-", color=GRIS, linewidth=0.6))
    ax_circ.axhline(0, color=GRIS, linewidth=0.8)
    ax_circ.axvline(0, color=GRIS, linewidth=0.8)
    ax_circ.set_xlim(-1.6, 1.6)
    ax_circ.set_ylim(-1.15, 1.15)
    ax_circ.set_aspect("equal")
    ax_circ.set_xlabel("Correlación con el componente 1")
    ax_circ.set_ylabel("Correlación con el componente 2")
    ax_circ.set_title("Círculo de correlaciones: hacia dónde aumenta cada variable", loc="left")
    fig.suptitle(f"Biplot PCA (k = {k}): {varianza.sum():.1%} de la varianza explicada", fontweight="bold")
    fig.tight_layout()
    guardar(fig, ruta)


def graficar_importancia(X_final, labels, k, ruta):
    # η² = varianza entre clústeres / varianza total, en la escala del modelo
    total = ((X_final - X_final.mean()) ** 2).sum()
    entre = sum((labels == c).sum() * (X_final[labels == c].mean() - X_final.mean()) ** 2
                for c in range(k))
    eta2 = (entre / total * 100).sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    barras = ax.barh([NOMBRES[v] for v in eta2.index], eta2.values, height=0.6, color=COLORES_CLUSTER[0])
    for barra, v in zip(barras, eta2.values):
        ax.annotate(f"{v:.1f} %", (barra.get_width(), barra.get_y() + barra.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", color=TINTA)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de la varianza de la variable explicada por el clúster (η²)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Peso de cada variable en la separación de los clústeres (k = {k})", loc="left")
    guardar(fig, ruta)
    return eta2.sort_values(ascending=False).rename("eta2_porcentaje")


def graficar_arbol(df, k, ruta, ruta_reglas):
    # Árbol de decisión sustituto: explica la asignación de K-Means con reglas sobre
    # las variables originales (sin transformar), fáciles de leer
    X = df[variables_cluster].fillna(df[variables_cluster].median())
    arbol = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=RANDOM_STATE)
    arbol.fit(X, df["cluster"])
    acierto = arbol.score(X, df["cluster"])
    clases = [f"Cluster {c}" for c in range(k)]

    fig, ax = plt.subplots(figsize=(26, 10))
    ax.grid(False)
    nodos = plot_tree(arbol, feature_names=[NOMBRES[v] for v in variables_cluster], class_names=clases,
                      filled=True, rounded=True, impurity=False, proportion=True, precision=2, fontsize=9,
                      ax=ax)
    traducciones = [("samples", "clientes"), ("value", "proporción"), ("class", "clúster"),
                    ("True", "Sí"), ("False", "No")]
    for nodo in nodos:
        texto = nodo.get_text()
        for ingles, espanol in traducciones:
            texto = texto.replace(ingles, espanol)
        nodo.set_text(texto)
        encontrado = re.search(r"clúster = Cluster (\d+)", nodo.get_text())
        caja = nodo.get_bbox_patch()
        if encontrado and caja is not None:
            caja.set_facecolor(color(int(encontrado.group(1))) + "40")
            caja.set_edgecolor(color(int(encontrado.group(1))))
    ax.set_title(f"Reglas que describen los clústeres (k = {k}): árbol de decisión de profundidad 3\n"
                 f"Reproduce el {acierto:.1%} de las asignaciones de K-Means "
                 "(\"proporción\" = parte de los clientes del nodo en cada clúster, en orden 0, 1, ...)", loc="left")
    guardar(fig, ruta)

    reglas = export_text(arbol, feature_names=variables_cluster, class_names=clases, decimals=2)
    Path(ruta_reglas).write_text(f"Acierto del árbol frente a K-Means: {acierto:.4f}\n\n{reglas}",
                                 encoding="utf-8")
    print(f"  {ruta_reglas}")


def caracterizar(df, X_final, kmeans, k, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    labels = kmeans.labels_
    print(f"\nCaracterización k = {k} -> {carpeta}")

    graficar_tamano(df, k, carpeta / "01_tamano_clusters.png")
    graficar_boxplots(df, k, carpeta / "02_boxplots_variables.png")
    pct, medianas = graficar_heatmap_perfil(df, k, carpeta / "03_heatmap_perfil.png")
    graficar_snake(X_final, labels, k, carpeta / "04_snake_plot.png")
    graficar_radar(df, k, carpeta / "05_radar_perfil.png")
    graficar_dispersion(df, k, carpeta / "06_dispersion_rfm.png")
    graficar_biplot(X_final, labels, kmeans, k, carpeta / "07_biplot_pca.png")
    eta2 = graficar_importancia(X_final, labels, k, carpeta / "08_importancia_variables.png")
    graficar_arbol(df, k, carpeta / "09_arbol_reglas.png", carpeta / "reglas_arbol.txt")
    for i, columna in enumerate(DESCRIPTIVAS, start=10):
        tabla = graficar_composicion(df, k, columna, carpeta / f"{i}_composicion_{columna}.png")
        tabla.round(2).to_csv(carpeta / f"composicion_{columna}.csv")

    graficar_silueta_clientes(X_final, labels, carpeta / "14_silueta_clientes.png")
    print(f"  {carpeta / '14_silueta_clientes.png'}")

    pct.round(1).add_prefix("percentil_").join(medianas.round(2).add_prefix("mediana_")) \
        .to_csv(carpeta / "perfil_percentiles.csv")
    eta2.round(2).to_csv(carpeta / "importancia_variables.csv")


# =============================================================================
# CLIENTES EN EL LÍMITE ENTRE DOS CLÚSTERES ("LIMBO")
# =============================================================================
def probabilidad_cambio(X_final, labels, k, n_submuestras=N_SUBMUESTRAS_LIMBO, fraccion=0.8):
    # Se reentrena con submuestras del 80 % y se predice a todos los clientes; las
    # etiquetas nuevas se alinean con las originales (algoritmo húngaro) y se cuenta
    # cuántas veces cada cliente termina en un clúster distinto
    rng = np.random.default_rng(RANDOM_STATE)
    cambios = np.zeros(len(X_final))
    for _ in range(n_submuestras):
        idx = rng.choice(len(X_final), size=int(fraccion * len(X_final)), replace=False)
        nuevas = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT) \
            .fit(X_final.iloc[idx]).predict(X_final)
        confusion = pd.crosstab(labels, nuevas).reindex(index=range(k), columns=range(k), fill_value=0)
        filas, columnas = linear_sum_assignment(-confusion.values)
        mapa = dict(zip(columnas, filas))
        cambios += np.array([mapa[v] for v in nuevas]) != labels
    return cambios / n_submuestras


def analizar_limbo(df, X_final, kmeans, k, carpeta, par=(0, 1)):
    carpeta.mkdir(parents=True, exist_ok=True)
    a, b = par
    labels = kmeans.labels_
    print(f"\nClientes en el límite entre Cluster {a} y Cluster {b} (k = {k}) -> {carpeta}")

    distancias = kmeans.transform(X_final)
    en_par = np.isin(labels, par)
    otro = np.where(labels == a, b, a)
    filas = np.arange(len(X_final))
    d_propio = distancias[filas, labels]
    d_otro = distancias[filas, otro]
    margen = (d_otro - d_propio) / d_otro
    segundo = np.argsort(distancias, axis=1)[:, 1]

    # Posición sobre el eje que une los dos centroides: 0 = centroide a, 1 = centroide b.
    # Entre dos centroides, el cliente está más cerca de a si y solo si t < 0,5
    ca, cb = kmeans.cluster_centers_[a], kmeans.cluster_centers_[b]
    t = (X_final.values - ca) @ (cb - ca) / np.sum((cb - ca) ** 2)

    silueta = silhouette_samples(X_final, labels)
    p_cambio = probabilidad_cambio(X_final, labels, k)

    limbo = en_par & (segundo == otro) & (margen < UMBRAL_LIMBO)
    nivel = np.where(margen < UMBRAL_LIMBO_CRITICO,
                     f"crítico (margen < {UMBRAL_LIMBO_CRITICO:.0%})",
                     f"cercano ({UMBRAL_LIMBO_CRITICO:.0%}-{UMBRAL_LIMBO:.0%})")

    tabla = pd.DataFrame({
        "cliente_id": df["cliente_id"],
        "cluster_actual": labels,
        "cluster_vecino": otro,
        "distancia_centroide_propio": d_propio,
        "distancia_centroide_vecino": d_otro,
        "margen_relativo": margen,
        "nivel_limbo": nivel,
        "posicion_eje_centroides": t,
        "silueta": silueta,
        "probabilidad_cambio_submuestras": p_cambio,
    })
    tabla = tabla.join(df[variables_cluster + list(DESCRIPTIVAS)])
    clientes_limbo = tabla[limbo].sort_values("margen_relativo")
    clientes_limbo.round(4).to_csv(carpeta / "clientes_limbo.csv", index=False)

    resumen = clientes_limbo.groupby(["cluster_actual", "nivel_limbo"]).agg(
        clientes=("cliente_id", "size"),
        margen_medio=("margen_relativo", "mean"),
        silueta_media=("silueta", "mean"),
        probabilidad_cambio_media=("probabilidad_cambio_submuestras", "mean"),
    ).round(4)
    resumen.to_csv(carpeta / "resumen_limbo.csv")
    print(f"Criterio: cliente del Cluster {a} o {b}, cuyo segundo centroide más cercano es el del otro, "
          f"con margen relativo < {UMBRAL_LIMBO:.0%}")
    print(resumen)

    grupos = [en_par & (labels == a) & ~limbo, limbo & (labels == a),
              limbo & (labels == b), en_par & (labels == b) & ~limbo]
    etiquetas = [f"Núcleo\nC{a}", f"Limbo\nen C{a}", f"Limbo\nen C{b}", f"Núcleo\nC{b}"]
    nombres_grupos = [f"Núcleo Cluster {a}", f"Limbo asignado a {a}", f"Limbo asignado a {b}",
                      f"Núcleo Cluster {b}"]
    colores = [color(a), color(a), color(b), color(b)]
    rayados = [None, "///", "///", None]
    conteos = [g.sum() for g in grupos]

    # 1. Histograma sobre el eje que une los centroides
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bins = np.arange(np.floor(t[en_par].min() * 10) / 10, t[en_par].max() + 0.05, 0.05)
    _, _, contenedores = ax.hist([t[g] for g in grupos], bins=bins, stacked=True, color=colores,
                                 edgecolor=SUPERFICIE, linewidth=0.6,
                                 label=[f"{e} ({n})" for e, n in zip(nombres_grupos, conteos)])
    for parche_grupo, rayado in zip(contenedores, rayados):
        for parche in parche_grupo:
            if rayado:
                parche.set_hatch(rayado)
                parche.set_alpha(0.75)
    # Rótulos de las líneas verticales por encima del área del gráfico
    arriba = blended_transform_factory(ax.transData, ax.transAxes)
    ax.axvline(0.5, color=TINTA, linestyle="--", linewidth=1.5)
    ax.text(0.5, 1.01, "frontera", transform=arriba, ha="center", va="bottom", fontsize=9, color=TINTA)
    for pos, c in [(0, a), (1, b)]:
        ax.axvline(pos, color=TINTA_SECUNDARIA, linewidth=1.2, linestyle=":")
        ax.text(pos, 1.01, f"centroide {c}", transform=arriba, ha="center", va="bottom", fontsize=9,
                color=TINTA)
    ax.set_xlabel(f"Posición del cliente sobre el eje que une el centroide {a} (0) con el centroide {b} (1)")
    ax.set_ylabel("Clientes")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Clientes de los clústeres {a} y {b} según su cercanía a cada centroide (k = {k})\n"
                 f"Rayado = clientes en el limbo (margen relativo < {UMBRAL_LIMBO:.0%}); "
                 f"a la izquierda de la frontera están más cerca del centroide {a}",
                 loc="left", pad=22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    guardar(fig, carpeta / "01_eje_entre_centroides.png")

    # 2. PCA con los clientes del limbo resaltados
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_final)
    centroides = pca.transform(pd.DataFrame(kmeans.cluster_centers_, columns=X_final.columns))
    fig, ax = plt.subplots(figsize=(12, 8))
    resto = ~en_par
    ax.scatter(X_pca[resto, 0], X_pca[resto, 1], s=8, color=GRIS, alpha=0.4, edgecolors="none",
               label="Otros clústeres")
    for c in par:
        m = en_par & (labels == c) & ~limbo
        ax.scatter(X_pca[m, 0], X_pca[m, 1], s=8, color=color(c), alpha=0.18, edgecolors="none",
                   label=f"Núcleo Cluster {c}")
    for c, marcador in [(a, "o"), (b, "s")]:
        m = limbo & (labels == c)
        ax.scatter(X_pca[m, 0], X_pca[m, 1], s=42, marker=marcador, color=color(c), edgecolors=TINTA,
                   linewidths=0.9, zorder=3, label=f"Limbo asignado a {c} ({m.sum()})")
    for c, (x, y) in enumerate(centroides):
        ax.scatter(x, y, s=260, marker="X", color=color(c) if c in par else GRIS, edgecolors=TINTA,
                   linewidths=1.5, zorder=4)
        ax.annotate(f"Centroide {c}", (x, y), xytext=(10, -16), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=TINTA,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SUPERFICIE, edgecolor="none", alpha=0.85))
    ax.set_xlabel(f"Componente principal 1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"Componente principal 2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"Clientes en el límite entre Cluster {a} y Cluster {b} (proyección PCA, k = {k})",
                 loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, markerscale=1.3)
    guardar(fig, carpeta / "02_pca_limbo.png")

    # 3. Perfil del limbo frente a los núcleos
    graficar_boxplots(df, k, carpeta / "03_perfil_limbo.png", grupos=grupos,
                      etiquetas=[f"{e}\n(n = {n})" for e, n in zip(etiquetas, conteos)],
                      colores=colores, rayados=rayados,
                      titulo=f"Perfil de los clientes en el limbo frente al núcleo de los clústeres {a} y {b}; "
                             "línea negra = mediana")

    # 4. Qué variables acercan a cada cliente del limbo al clúster vecino. Aporte de la
    #    variable j a d_propio² - d_vecino²: positivo = esa variable lo empuja al vecino
    centros = kmeans.cluster_centers_
    aporte = (X_final.values - centros[labels]) ** 2 - (X_final.values - centros[otro]) ** 2
    aporte = pd.DataFrame(aporte, columns=variables_cluster)
    medias = pd.DataFrame({c: aporte[limbo & (labels == c)].mean() for c in par})
    medias.round(4).to_csv(carpeta / "aporte_variables_limbo.csv")

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(variables_cluster))
    alto = 0.38
    for i, c in enumerate(par):
        destino = b if c == a else a
        ax.barh(y + (i - 0.5) * alto, medias[c], height=alto, color=color(c), edgecolor=SUPERFICIE,
                linewidth=2, label=f"Limbo asignado a {c} (vecino: {destino})")
    ax.axvline(0, color=TINTA, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([NOMBRES[v] for v in variables_cluster])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Aporte medio a (distancia² al centroide propio − distancia² al vecino)\n"
                  "← lo retiene en su clúster  |  lo acerca al clúster vecino →")
    ax.set_title("Variables que acercan a los clientes del limbo al clúster vecino", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    guardar(fig, carpeta / "04_variables_que_acercan.png")

    # 5. Margen frente a inestabilidad en submuestras
    fig, ax = plt.subplots(figsize=(11, 6))
    rng = np.random.default_rng(RANDOM_STATE)
    for c, marcador in [(a, "o"), (b, "s")]:
        m = en_par & (labels == c)
        ax.scatter(margen[m] * 100, p_cambio[m] * 100 + rng.uniform(-0.6, 0.6, m.sum()), s=14,
                   marker=marcador, color=color(c), alpha=0.5, edgecolors="none", label=f"Cluster {c}")
    ax.axvline(UMBRAL_LIMBO * 100, color=TINTA, linestyle="--", linewidth=1.2)
    ax.axvline(UMBRAL_LIMBO_CRITICO * 100, color=TINTA, linestyle=":", linewidth=1.2)
    ax.annotate(f"limbo < {UMBRAL_LIMBO:.0%}", (UMBRAL_LIMBO * 100, 100), xytext=(4, -2),
                textcoords="offset points", va="top", fontsize=9, color=TINTA)
    ax.annotate(f"crítico < {UMBRAL_LIMBO_CRITICO:.0%}", (UMBRAL_LIMBO_CRITICO * 100, 92), xytext=(4, -2),
                textcoords="offset points", va="top", fontsize=9, color=TINTA)
    ax.set_xlabel("Margen relativo (%): cuánto más cerca está de su centroide que del vecino")
    ax.set_ylabel(f"% de submuestras en que cambia de clúster\n({N_SUBMUESTRAS_LIMBO} submuestras del 80 %)")
    ax.set_ylim(-3, 103)
    ax.set_title(f"Cercanía a la frontera frente a inestabilidad de la asignación (Clusters {a} y {b})",
                 loc="left")
    ax.legend(loc="upper right")
    guardar(fig, carpeta / "05_margen_vs_inestabilidad.png")


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def leer_par(texto):
    a, b = (int(x) for x in texto.split("-"))
    return a, b


def main():
    parser = argparse.ArgumentParser(description="Caracterización gráfica de clústeres de K-Means")
    parser.add_argument("--salida", default=str(Path(__file__).resolve().parent / "resultados"),
                        help="Carpeta de resultados (por defecto: caracterizacion/resultados)")
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3], help="Valores de k a caracterizar")
    parser.add_argument("--k-limbo", type=int, default=3,
                        help="k en el que se buscan clientes en el límite entre clústeres")
    parser.add_argument("--limbo", type=leer_par, nargs="+", default=[(0, 1), (1, 2)], metavar="A-B",
                        help="Pares de clústeres para el análisis del limbo (por defecto: 0-1 1-2)")
    parser.add_argument("--referencia", default=str(RAIZ / "src" / "Resultados_K-2-k-3"),
                        help="Carpeta con k<k>/clientes_cluster.csv de main.py para comprobar "
                             "que las etiquetas coinciden")
    parser.add_argument("--datos", help="Archivo .pkl con el resultado de la consulta: se lee si existe "
                                        "y se crea si no, para no repetir el SQL")
    args = parser.parse_args()

    estilo()
    salida = Path(args.salida)

    if args.datos and Path(args.datos).exists():
        df = pd.read_pickle(args.datos)
        print(f"Datos leídos de {args.datos}: {df.shape}")
    else:
        df = cargar_datos(crear_engine())
        if args.datos:
            df.to_pickle(args.datos)
    X_final = preparar_matriz(df)

    for k in sorted(set(args.k) | {args.k_limbo}):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(X_final)
        df_k = df.copy()
        df_k["cluster"] = kmeans.labels_

        previo = Path(args.referencia) / f"k{k}" / "clientes_cluster.csv"
        if previo.exists():
            comparacion = df_k[["cliente_id", "cluster"]].merge(pd.read_csv(previo), on="cliente_id")
            ari = adjusted_rand_score(comparacion["cluster_x"], comparacion["cluster_y"])
            print(f"\nk = {k}: ARI frente a {previo} = {ari:.4f}")

        if k in args.k:
            caracterizar(df_k, X_final, kmeans, k, salida / "graficos_y_reportes" / f"k{k}")
        if k == args.k_limbo:
            for a, b in args.limbo:
                analizar_limbo(df_k, X_final, kmeans, k, salida / "limbo" / f"cluster_{a}_cluster_{b}",
                               par=(a, b))


if __name__ == "__main__":
    main()
