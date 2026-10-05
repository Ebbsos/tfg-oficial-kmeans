# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Thesis project (TFG) for customer segmentation with K-Means. Everything is written in Spanish (variable names, comments, plot labels, printed output). Keep new code in Spanish to match. There is no package structure, test suite, or linter.

- [main.py](main.py) is the **authoritative version** of the analysis. It is organized in functions (`cargar_datos`, `preparar_matriz`, `evaluar_k`, `perfilar_clusters`, `graficar_*`) called from `main()`.
- [elbow-kmeans-silluethe.ipynb](elbow-kmeans-silluethe.ipynb) is the original notebook and reflects the **earlier methodology** (see below). It also has an **Apriori per cluster** section (`mlxtend`) that was **intentionally removed from `main.py`**. Do not add Apriori or `mlxtend` back to the script or to `requirements.txt`.

Differences between the script and the notebook (the script's choices are the deliberate ones):

| | `main.py` | notebook |
|---|---|---|
| Clustering features | 8 numeric only; canal/localidad only describe clusters | 8 numeric + 44 one-hot dummies of canal/localidad |
| Localidad/canal per client | highest spend (`localidad_top`/`canal_top` CTEs) | `MAX(id)` |
| Null percentages | imputed with the median, fails if NaN remains | not handled |
| Cluster profiles | medians | means |
| Output | files in `--salida` dir, no windows | `plt.show()`, image to `~/Downloads` |

## Running

- Install: `pip install -r requirements.txt` (minimum versions only, nothing pinned). A local `venv/` (git-ignored) has everything installed: `venv/Scripts/python.exe main.py`.
- Credentials: `DB_USER` and `DB_PASSWORD` are required, read from the environment or a git-ignored `.env` (template in [.env.example](.env.example)). `DB_HOST`, `DB_PORT` and `DB_NAME` default to `localhost`, `3306` and `renacertfg`. Needs a **local MySQL** database; nothing runs without it.
- Run: `python main.py --salida src/ejecucion_N`. Optional `--k N` fixes the final number of clusters instead of the best silhouette; elbow and silhouette are still computed and reported for every k. A full run takes about 6 minutes, mostly the SQL query. On Windows set `PYTHONIOENCODING=utf-8` when redirecting output.
- Outputs in the `--salida` folder: `01_codo.png`, `02_silhouette.png`, `03_pca_clusters.png`, `04_xfinal_muestra.png`, `05_estabilidad_ari.png`, `06_estabilidad_jaccard.png`, `metricas_k.csv`, `resumen_clusters.csv`, `detalle_clusters.csv`, `silueta_por_cluster.csv`, `estabilidad_*.csv` (ARI for k=2, 3, 4 and the final k across 10 seeds, 20 subsamples of 80% without replacement predicted on all clients, and 100 Hennig-style bootstrap resamples compared only on the distinct clients drawn; plus per-cluster max Jaccard), and `clientes_cluster.csv` (`cliente_id` → cluster, useful to compare runs with `adjusted_rand_score`).
- `src/ejecucion_*` folders hold results of past runs for comparison. `src/ejecucion_1` was produced with the notebook methodology and only contains figures.

## Pipeline

1. **Data extraction (SQL)**: one CTE query against the view `vw_fact_ventas_tipada`, joined with `dim_productos`, `dim_categorias`, `dim_marcas`, `dim_localidades`, `dim_canales`. The period is 2022-01-01 to 2024-12-31, and recency is measured against 2024-12-31. It builds one row per `cliente_id` (3757 clients) with RFM features (recencia, frecuencia_compra = distinct documento-serie-numero, monto_total), variety counts, and the dominant category and brand with their % share of spend.
2. **Feature matrix `X_final`**: median imputation of the two percentage columns, `log1p` on the count/amount features, then `RobustScaler` on the 8 features in `variables_cluster`.
3. **Choosing k**: one loop over k=1..10 computes inertia and the silhouette score (sampled above 10,000 clients). `KneeLocator` gives the elbow `k_optimo`. **The final model is the k with the best silhouette (`best_k`)**, reused from the loop rather than refit. The elbow k is only reported.
4. **Profiling**: sizes, medians, and the most frequent canal/localidad/categoría/marca per cluster, plus a PCA 2-D plot whose labels show the explained variance.
5. **Export**: a table image of the first rows of `X_final`.

Keep `random_state=42` and `n_init=20` (`RANDOM_STATE`, `N_INIT`) wherever KMeans is used so the results stay reproducible.

## Known data notes

- `dim_canales` has a double-encoded name (`PERFUMERΓA`) next to `PERFUMERIA B`. It only affects labels, since canal is not a clustering feature.
- Silhouette values are only comparable between runs if they are computed in the same feature space.
