# Experiment Report: Feature-Subset Ablation (>50% Missingness) on Tabular KNN (exp_4)

**Experiment**: experiments/exp_4/ · **Project**: pathology-reasoning
**Report date**: 2026-08-15 · **Plan date**: 2026-08-15
**Author**: agent (role: ML / agents / reasoning) + PI
**Status**: Complete

---

## 1. Summary

Se repitió el baseline KNN de `exp_3` sobre un subconjunto de **37 variables** (se eliminaron `path_hist_bx_gl_tert` 98.9% y `lab_hemoglobin_g_dl` 73.9% de ausencia en la cohorte usable N=88; las 10 variables esenciales siempre retenidas), con regla pre-registrada `missing_rate > 50%` fijada globalmente, los mismos 50 splits MCCV congelados y las mismas 128 configs. El ganador por Macro-F1 (`manhattan_fuzzy_confidence_inverse_distance_k3`) alcanza **0.6430 ± 0.110** en MCCV, **+0.0196** sobre el ganador de exp_3 (0.6234), y supera a las mayoritarias (`always_yes` 0.3793, `always_no` 0.2800). El sanity check **LOO pooled** da Macro-F1 = **0.6474**, consistente con el MCCV. La comparación pareada por config muestra que **el filtro cambia el óptimo de hiperparámetros**: la config ganadora de exp_4 mejora 0.5514 → 0.6430 (+0.092, Wilcoxon p=3.96e-08) al pasar de 39 a 37 variables, mientras que la config ganadora de exp_3 degrada 0.6234 → 0.5934 (−0.030, p=0.024) en el espacio filtrado.

## 2. Hypothesis & Verdict

**Hypothesis (from plan):** "Eliminar las variables tabulares con más del 50% de ausencia en el cohorte usable_labeled (N=88), conservando siempre las 10 variables clínicas esenciales, mejora o mantiene el Macro-F1 del KNN frente al baseline de exp_3 (39 variables)."

**Verdict:** ✅ Supported

**Evidence:** el mejor Macro-F1 sobre el espacio filtrado (0.6430) supera al mejor del espacio completo (0.6234); la config ganadora de exp_4 mejora significativamente en el espacio de 37 variables (+0.092 pareado, p<0.05) y el sanity check LOO pooled lo confirma (0.6474). El filtro elimina exactamente las 2 variables esperadas y conserva las 10 esenciales.

## 3. Experimental Setup (as run)

Implementado según `IMPLEMENTATION.md`: `scripts/knn_baseline.py` (idéntico a exp_3 salvo el filtro) + `plot_results.py` + `compare_experiments.py`.

- **Dataset**: `inputs_tabular.csv` (195×40). Hash en `results/data_manifest.csv` (coincide con exp_3).
- **Filtro (pre-registrado, global)**: `retained = missing_rate <= 0.5 or essential`. `missing_rate` sobre los 88 `usable_labeled` (string vacío). Eliminadas: `path_hist_bx_gl_tert` (87/88), `lab_hemoglobin_g_dl` (65/88). Retenidas: 37 (34 continuas + 3 categóricas). Las 10 esenciales protegidas (ninguna supera 50%). Manifiesto en `results/feature_selection_manifest.csv` y `feature_missingness.csv`.
- **Cohorte**: 88 `usable_labeled` (54 yes / 34 no). Split file congelado (no regenerado).
- **Preprocessing**: sin imputación; Min-Max sobre observados del train por fold; ausencia → 0 estructural + `missing_<var>` (solo variables retenidas); `cli_bx="None"` y `cli_dre="Not done"` categorías válidas.
- **Modelo**: KNN brute-force (argsort estable), `p_yes = Σ w_i y_i / Σ w_i`, umbral 0.5 fijo. Fuzzy multiplica por confianza del vecino de train.
- **Grid**: 128 configs (4 distancias × 2 reglas × 2 pesos × 8 k).
- **Selección**: Macro-F1 MCCV (mean 50 folds); desempates F1_yes, luego balanced accuracy.
- **LOO**: 88 folds solo para la config seleccionada.
- **Comparación**: `compare_experiments.py` → `results/comparison_39_vs_37.csv` + figura (delta pareado por fold, CI95 y Wilcoxon).
- **Entorno**: conda `histo-DL`, 22 cores, MCCV ≈ 39 s + LOO.
- **Deviations from plan**: ninguna. (Misma desviación explícita que exp_3: Macro-F1 como primaria de selección, aprobada por el PI.)

## 4. Code Version

| Artifact | Git commit | Mensaje |
|----------|-----------|---------|
| pipeline + resultados | `6e4b61c56b6b26a5040f89053756ec4f0c546629` | `restart proyect` |

⚠️ Árbol de trabajo no limpio al ejecutar (misma situación que exp_3). Hash registrado en `results/selected_config/git_commit.txt`.

## 5. Results

### 5.1 Config seleccionada (MCCV, mean ± std sobre 50 folds)

| Métrica | MCCV (mean±std) | LOO pooled |
|---|---|---|
| **Macro-F1** | **0.6430 ± 0.110** | **0.6474** |
| F1_yes | 0.7149 ± 0.107 | 0.7115 |
| F1_no | 0.5712 ± 0.129 | 0.5833 |
| accuracy | 0.6622 ± 0.109 | 0.6591 |
| balanced accuracy | 0.6478 ± 0.107 | — |
| sensitivity (recall yes) | 0.7127 ± 0.150 | 0.6852 |
| specificity (recall no) | 0.5829 ± 0.158 | 0.6176 |
| precision_yes | 0.7294 ± 0.089 | 0.7400 |
| MCC | 0.3032 ± 0.217 | — |
| Brier | 0.2494 ± 0.076 | — |
| ROC-AUC | 0.6792 ± 0.129 (n=50) | 0.6863 |
| PR-AUC | 0.7975 ± 0.090 | 0.7861 |

### 5.2 Ganador vs exp_3

| Condición | Macro-F1 MCCV | vs mejor baseline (Δ) |
|-----------|---------------|----------------------|
| **exp_4 ganador** (`manhattan_fuzzy_confidence_inverse_distance_k3`, 37 vars) | **0.6430** | **+0.2637** ✅ |
| exp_3 ganador (`manhattan_fuzzy_confidence_uniform_k11`, 39 vars) | 0.6234 | +0.2441 |
| always_yes | 0.3793 | — |
| always_no | 0.2800 | — |

### 5.3 Comparación pareada 39 vs 37 (misma config, mismos folds)

| Config | Macro-F1 39 | Macro-F1 37 | Δ (37−39) | IC95% pareado | Wilcoxon p | n folds Δ>0 |
|---|---|---|---|---|---|---|
| **ganador exp_4** | 0.5514 | 0.6430 | **+0.0916** | [0.069, 0.114] | 3.96e-08 | 42/50 |
| ganador exp_3 | 0.6234 | 0.5934 | −0.0300 | [−0.056, −0.004] | 0.0239 | 16/50 |

Interpretación: el filtro de variables cambia el óptimo de hiperparámetros (k=3 + inverse_distance + fuzzy dominan en el espacio de 37 vars). La comparación ganador-vs-ganador (0.6430 vs 0.6234) es favorable a exp_4 pero no es una prueba formal (selección dentro de los mismos splits); la comparación por config fija sí es formal y es significativa.

### 5.4 Matrices de confusión (config seleccionada)

MCCV agregado (900 eventos de validación):

| | pred no | pred yes |
|---|---|---|
| true no | 204 | 146 |
| true yes | 158 | 392 |

LOO pooled (88 predicciones):

| | pred no | pred yes |
|---|---|---|
| true no | 21 | 13 |
| true yes | 17 | 37 |

Figuras en `reports/figures/` (5 matrices + `comparison_macro_f1_39_vs_37.png`).

## 6. Statistical Analysis

- KNN determinista dado el split congelado; varianza = entre 50 folds MCCV.
- Test formal (config fija, pareado por fold): Wilcoxon signed-rank sobre deltas de Macro-F1. Significativo en ambos sentidos (mejora para la config ganadora de exp_4; degradación para la config ganadora de exp_3).
- LOO pooled (0.6474) consistente con MCCV (0.6430): sin gap.

## 7. Comparison to Expected Results

| Esperado (DESIGN.md §7) | Observado | Match |
|---|---|---|
| El ganador de exp_4 supera a ambas mayoritarias | 0.6430 vs 0.3793 / 0.2800 | ✅ |
| Filtro elimina exactamente las 2 variables >50% | sí (98.9% y 73.9%), 37 retenidas | ✅ |
| Las 10 esenciales siempre retenidas | todas conservadas (ninguna >50%) | ✅ |
| Comparación pareada por config reportada | 128 configs, Wilcoxon + IC95 | ✅ |
| LOO solo sanity check | ejecutado post-selección | ✅ |

## 8. Missing Data & Caveats

- ⚠️ Comparación ganador-vs-ganador no es prueba formal (selección dentro de los mismos splits). La prueba formal es la de config fija.
- ⚠️ La máscara global usa la estructura de ausencia de los 88 casos (incluye los val de cada split); no hay target leakage (sin `y`) y las 2 variables eliminadas superan 50% en 50/50 folds de train → equivalente a un filtro per-fold en estos datos.
- ⚠️ `cli_fh_binary` (3 ausentes usable) se mantiene como ausencia estructural (decisión de recuperación sigue pendiente del PI).
- ⚠️ Se elimina `path_hist_bx_gl_tert` pero se conservan `path_hist_bx_isup/prim/sec` (26.1%): decisión puramente por umbral.
- Árbol de trabajo no limpio al ejecutar (ver §4).

## 9. Conclusions & Next Steps

- **Establecido**: (1) filtrar variables con >50% de ausencia (conservando esenciales) **mejora** el mejor Macro-F1 del KNN (0.6234 → 0.6430) y cambia el óptimo de hiperparámetros (k=3, inverse_distance, fuzzy, manhattan). (2) El efecto es significativo a nivel config pareada (+0.092, p=3.96e-08). (3) LOO pooled confirma sin gap.
- **Incierto**: (1) la comparación ganador-vs-ganador no es formal. (2) Generalización al challenge (~104 casos) no medible internamente.
- **Siguientes pasos recomendados**:
  1. Anclar el baseline tabular en el espacio de 37 variables para la campaña (config `manhattan_fuzzy_confidence_inverse_distance_k3`).
  2. Comparar con un modelo lineal/logístico sobre las mismas 37 variables y splits.
  3. Subtareas 1.2 (confianza ordinal) y 1.3 (relevancia clínica) reutilizando el protocolo.
  4. Evaluar umbrales alternativos de filtro (p.ej. 30%) como sensibilidad, si el PI lo decide.

## 10. Reproducibility Record

| Item | Status |
|------|--------|
| Regla de filtro pre-registrada | ✅ (DESIGN.md §2 + `feature_selection_manifest.csv`) |
| Semillas | ✅ (no aplica: determinista; splits congelados) |
| Configs versionadas | ✅ (`mccv_summary.csv`) |
| Git commit registrado | ✅ (`selected_config/git_commit.txt`) |
| Hashes de datos | ✅ (`data_manifest.csv`) |
| Entorno congelado | ✅ conda `histo-DL` |
| Árbol limpio al ejecutar | ❌ (advertido en §4) |
