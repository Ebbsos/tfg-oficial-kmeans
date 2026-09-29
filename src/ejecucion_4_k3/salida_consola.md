# Salida de consola: ejecución 4 (k = 3)

Comando ejecutado:

```bash
python main.py --k 3 --salida src/ejecucion_4_k3
```

La salida original, sin formato, está en [salida_consola.log](salida_consola.log). Los valores son idénticos a los de [ejecucion_3](../ejecucion_3/).

---

## 1. Datos

- **Dimensión original:** 3757 clientes × 15 columnas.
- **Valores nulos:** ninguna columna tiene nulos, así que no se imputó ningún valor.
- **Dimensión final de `X_final`:** 3757 clientes × 8 variables.

Primeras filas de `X_final` (después de `log1p` y `RobustScaler`):

| | recencia | frecuencia_compra | monto_total | variedad_productos | cantidad_categorias_distintas | cantidad_marcas_distintas | porcentaje_monto_categoria_top | porcentaje_monto_marca_top |
|---|---|---|---|---|---|---|---|---|
| 0 | -0,389683 | 0,681632 | 0,781217 | 0,823787 | 0,633761 | 1,594008 | 0,081345 | -0,085176 |
| 1 | 2,096093 | -0,561478 | -0,312060 | -0,419591 | -0,239823 | -0,331094 | -0,075088 | 0,168000 |
| 2 | -0,389683 | 0,573556 | 0,392595 | 0,721382 | 0,633761 | 1,594008 | -0,883066 | -0,906824 |
| 3 | 0,000000 | 0,319744 | 0,183955 | 0,354003 | 0,224340 | 0,831394 | 0,281580 | 0,008941 |
| 4 | 0,372322 | -0,171590 | -0,010107 | -0,466067 | 0,000000 | 0,153897 | 0,929996 | 0,995765 |

## 2. Método del codo y coeficiente de silueta

| k | Inercia | Silueta |
|---|---|---|
| 1 | 36 547,98 | — |
| 2 | 16 984,18 | **0,644944** |
| 3 | **11 237,97** | 0,429661 |
| 4 | 9 584,98 | 0,374912 |
| 5 | 8 309,14 | 0,254101 |
| 6 | 7 259,45 | 0,269064 |
| 7 | 6 567,70 | 0,268226 |
| 8 | 6 083,26 | 0,265031 |
| 9 | 5 614,55 | 0,268672 |
| 10 | 5 265,17 | 0,260460 |

- **k óptimo según el codo (`KneeLocator`):** 3.
- **Mejor k según la silueta:** 2 (0,644944).
- **k usado en esta ejecución:** 3, fijado manualmente con `--k 3`. Su silueta es 0,429661.

> Nota: la consola también imprime *"el modelo final usa k = 2 (silhouette)"*. Ese mensaje sale antes de que se aplique `--k`, así que no refleja el modelo usado. El modelo final de esta ejecución es el de **k = 3**.

Figuras: [01_codo.png](01_codo.png) y [02_silhouette.png](02_silhouette.png).

## 3. Perfil de los clústeres

### Medianas y valores predominantes

| Clúster | Clientes | % | Recencia (días) | Frecuencia | Monto total | Variedad de productos | Categorías distintas | Marcas distintas | % monto categoría top | % monto marca top |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2461 | 65,50 | 3 | 180 | 19 024 939,40 | 202 | 14 | 22 | 59,59 | 39,29 |
| 1 | 1023 | 27,23 | 232 | 34 | 1 901 006,19 | 51 | 10 | 14 | 48,35 | 33,40 |
| 2 | 273 | 7,27 | 147 | 2 | 159 718,70 | 4 | 2 | 3 | 81,67 | 68,47 |

Los valores predominantes son los mismos en los tres clústeres:

| Canal | Localidad | Categoría | Marca |
|---|---|---|---|
| DESPENSAS B | Ciudad del Este | LAUNDRY | OMO |

### Estadísticas detalladas (media, mediana, mínimo y máximo)

| Variable | Clúster | Media | Mediana | Mín. | Máx. |
|---|---|---|---|---|---|
| recencia | 0 | 16,48 | 3 | 1 | 925 |
| | 1 | 341,31 | 232 | 1 | 1071 |
| | 2 | 347,36 | 147 | 1 | 1090 |
| frecuencia_compra | 0 | 182,73 | 180 | 37 | 1205 |
| | 1 | 41,62 | 34 | 2 | 163 |
| | 2 | 5,02 | 2 | 1 | 152 |
| monto_total | 0 | 78 305 798,23 | 19 024 939,40 | 1 806 241,11 | 5,543295 × 10⁹ |
| | 1 | 4 900 600,05 | 1 901 006,19 | 62 674,52 | 3,802741 × 10⁸ |
| | 2 | 53 526 789,39 | 159 718,70 | 4 499,06 | 1,002358 × 10¹⁰ |
| variedad_productos | 0 | 233,29 | 202 | 55 | 1201 |
| | 1 | 55,56 | 51 | 7 | 264 |
| | 2 | 6,18 | 4 | 1 | 84 |
| cantidad_categorias_distintas | 0 | 14,20 | 14 | 9 | 19 |
| | 1 | 9,98 | 10 | 3 | 15 |
| | 2 | 2,57 | 2 | 1 | 6 |
| cantidad_marcas_distintas | 0 | 22,21 | 22 | 14 | 47 |
| | 1 | 14,17 | 14 | 6 | 22 |
| | 2 | 3,13 | 3 | 1 | 8 |
| porcentaje_monto_categoria_top | 0 | 59,22 | 59,59 | 18,93 | 99,54 |
| | 1 | 50,74 | 48,35 | 18,08 | 99,68 |
| | 2 | 77,91 | 81,67 | 31,61 | 100,00 |
| porcentaje_monto_marca_top | 0 | 40,53 | 39,29 | 11,66 | 99,54 |
| | 1 | 38,13 | 33,40 | 13,02 | 99,68 |
| | 2 | 70,01 | 68,47 | 26,65 | 100,00 |

## 4. PCA

| Componente | Varianza explicada |
|---|---|
| PC1 | 76,60 % |
| PC2 | 12,00 % |
| **Total** | **88,59 %** |

Figura: [03_pca_clusters.png](03_pca_clusters.png).

## 5. Silueta por clúster

| Clúster | Clientes | Silueta media | Silueta mediana | % con silueta negativa |
|---|---|---|---|---|
| 0 | 2461 | 0,5145 | 0,5509 | 0,00 % |
| 1 | 1023 | 0,2231 | 0,2426 | 12,32 % |
| 2 | 273 | 0,4387 | 0,5345 | 3,30 % |

## 6. Estabilidad: ARI

Cada partición se compara con el modelo de referencia (`random_state=42`, `n_init=20`, entrenado con los 3757 clientes). Hay dos pruebas:
- **Semilla:** 10 reentrenamientos con `random_state` de 1 a 10, sobre todos los clientes.
- **Submuestra:** 20 submuestras aleatorias sin reemplazo del 80 % (3005 clientes). En cada una se reentrena el modelo y se asigna a los 3757 clientes.

| k | Prueba | ARI media | Desv. estándar | ARI mín. | ARI máx. | Repeticiones |
|---|---|---|---|---|---|---|
| 2 | semilla | 1,0000 | 0,0000 | 1,0000 | 1,0000 | 10 |
| 2 | submuestra | 0,9938 | 0,0090 | 0,9752 | 1,0000 | 20 |
| **3** | **semilla** | **1,0000** | **0,0000** | **1,0000** | **1,0000** | **10** |
| **3** | **submuestra** | **0,9880** | **0,0100** | **0,9687** | **0,9990** | **20** |
| 4 | semilla | 0,9989 | 0,0017 | 0,9944 | 1,0000 | 10 |
| 4 | submuestra | 0,9710 | 0,0229 | 0,9168 | 0,9923 | 20 |

## 7. Estabilidad por clúster: índice de Jaccard (k = 3)

Para cada clúster de referencia se toma el máximo Jaccard frente a los clústeres de la partición alternativa.

| Prueba | Clúster 0: media | Clúster 0: mín. | Clúster 1: media | Clúster 1: mín. | Clúster 2: media | Clúster 2: mín. |
|---|---|---|---|---|---|---|
| semilla | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| submuestra | 0,9958 | 0,9899 | 0,9852 | 0,9578 | 0,9822 | 0,9267 |

## Anexo: detalle por repetición (k = 3)

Fuente: [estabilidad_detalle.csv](estabilidad_detalle.csv). Las 10 repeticiones de la prueba de semillas dan ARI = 1 y Jaccard = 1 en los tres clústeres. Las 20 submuestras:

| Submuestra | ARI | Jaccard clúster 0 | Jaccard clúster 1 | Jaccard clúster 2 |
|---|---|---|---|---|
| 1 | 0,9708 | 0,9899 | 0,9633 | 0,9524 |
| 2 | 0,9713 | 0,9911 | 0,9597 | 0,9267 |
| 3 | 0,9687 | 0,9899 | 0,9578 | 0,9304 |
| 4 | 0,9872 | 0,9947 | 0,9873 | 1,0000 |
| 5 | 0,9951 | 0,9980 | 0,9951 | 1,0000 |
| 6 | 0,9961 | 0,9984 | 0,9961 | 1,0000 |
| 7 | 0,9961 | 0,9984 | 0,9961 | 1,0000 |
| 8 | 0,9816 | 0,9939 | 0,9758 | 0,9634 |
| 9 | 0,9918 | 0,9967 | 0,9913 | 0,9964 |
| 10 | 0,9961 | 0,9984 | 0,9961 | 1,0000 |
| 11 | 0,9941 | 0,9976 | 0,9942 | 1,0000 |
| 12 | 0,9911 | 0,9963 | 0,9913 | 1,0000 |
| 13 | 0,9941 | 0,9976 | 0,9942 | 1,0000 |
| 14 | 0,9932 | 0,9980 | 0,9903 | 0,9817 |
| 15 | 0,9757 | 0,9919 | 0,9681 | 0,9524 |
| 16 | 0,9990 | 0,9996 | 0,9990 | 1,0000 |
| 17 | 0,9811 | 0,9931 | 0,9776 | 0,9780 |
| 18 | 0,9970 | 0,9988 | 0,9971 | 1,0000 |
| 19 | 0,9980 | 0,9992 | 0,9980 | 1,0000 |
| 20 | 0,9816 | 0,9939 | 0,9758 | 0,9634 |
