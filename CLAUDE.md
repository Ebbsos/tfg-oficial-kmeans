# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Thesis project (TFG) for customer segmentation with K-Means. Everything is written in Spanish (variable names, comments, plot labels, printed output). Keep new code in Spanish to match. There is no package structure, test suite, or linter.

There are two entry points with the same logic:

- [main.py](main.py): the script version and the main one to edit. It is a linear top-level script with no functions except where the notebook had them, and state is kept in module globals.
- [elbow-kmeans-silluethe.ipynb](elbow-kmeans-silluethe.ipynb): the original notebook. It has one extra section at the end, **Apriori per cluster** (association rules with `mlxtend`), that was **intentionally removed from `main.py`**. Do not add Apriori or `mlxtend` back to the script or to `requirements.txt`.

When the analysis logic changes in one entry point, mirror the change in the other (except for Apriori).

`xfinal_muestra.png` is a sample of the `X_final` matrix rendered as a table for the thesis document. Both entry points write it to `~/Downloads/xfinal_muestra.png`, not to the repo.

## Running

- Install: `pip install -r requirements.txt` (minimum versions only, nothing pinned). A local `venv/` may exist; it is git-ignored.
- Run the script: `python main.py`. Each `plt.show()` blocks until its window is closed (elbow, silhouette, PCA, then the X_final table image).
- Run the notebook headless: `jupyter nbconvert --to notebook --execute elbow-kmeans-silluethe.ipynb --output executed.ipynb`. The notebook kernel was the conda `base` environment (Python 3.13), and it also needs `mlxtend` for its Apriori section.
- Both entry points need a **local MySQL** database `renacertfg`, with defaults `root/root@localhost:3306`. In `main.py` these can be overridden with `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`. The notebook hard-codes them. Nothing past the connection step runs without the database.

## Pipeline (order matters; later steps depend on globals from earlier ones)

1. **Data extraction (SQL)**: one large CTE query against the view `vw_fact_ventas_tipada`, joined with `dim_productos`, `dim_categorias`, `dim_marcas`, `dim_localidades`, `dim_canales`. The period is 2022-01-01 to 2024-12-31, and recency is measured against 2024-12-31. It builds one row per `cliente_id` with RFM features (recencia, frecuencia_compra = distinct documento-serie-numero, monto_total), variety counts, and the dominant category and brand with their % share of spend. The result is the `df` DataFrame.
2. **Feature matrix `X_final`**: `log1p` on the count/amount features, `RobustScaler` on the 8 numeric features (`variables_cluster`), then concatenation with one-hot dummies of `canal` and `localidad` (NaN is filled with `SIN_CANAL`/`SIN_LOCALIDAD`).
3. **Choosing k**: the elbow method (k=1..10, `KneeLocator` gives `k_optimo`) and the silhouette score (k=2..10, which gives `best_k`). **The final model uses `best_k` (silhouette), not the elbow k.** The elbow step is only for the plot and the report.
4. **K-Means**: `KMeans(n_clusters=best_k, random_state=42, n_init=20)`. Labels are stored in `df["cluster"]`, followed by per-cluster profiling tables (means, dominant canal/localidad/categoría/marca, sizes and percentages) and a 2-D PCA scatter plot with the centroids projected.
5. **Export**: a table image of the first 8 rows × 10 columns of `X_final`.

Keep `random_state=42` and `n_init=20` wherever KMeans is used so the results stay reproducible across the elbow, silhouette, and final model.

## Notebook-specific notes

- Section headers are **code cells containing only a comment** (e.g. `#SILHOUETTE`, `#MODELO K-MEANS`), not markdown cells. In `main.py` they are `# ===` comment banners.
- The notebook has several overlapping variants of the summary table (`tabla_resumen` is redefined) and uses `display()` and `.style.background_gradient`. `main.py` keeps only the final `tabla_resumen` and uses `print()`.
