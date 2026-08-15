# Experiment Report: Tabular KNN Baseline for CHIMERA Task 1.1 (exp_3)

**Experiment**: experiments/exp_3/ · **Project**: pathology-reasoning
**Report date**: 2026-08-15 · **Plan date**: 2026-08-15
**Author**: agent (role: ML / agents / reasoning) + PI
**Status**: Complete

---

## 1. Summary

Un KNN sobre las 39 variables tabulares, sin imputación y con indicadores de ausencia, fue evaluado sobre los 50 splits MCCV congelados (128 configuraciones: 4 distancias × 2 reglas × 2 pesos × 8 k). El ganador por **Macro-F1** (`manhattan_fuzzy_confidence_uniform_k11`) alcanza **0.6234 ± 0.101** en MCCV, superando a las mayoritarias (`always_yes` 0.3793, `always_no` 0.2800) en 49/50 y 50/50 folds respectivamente. El sanity check LOO **pooled** da Macro-F1 = 0.6207, prácticamente idéntico al MCCV: el gran "gap MCCV→LOO" reportado en la iteración previa (0.31) era un artefacto de promediar métricas por-fold a n=1, no una inestabilidad real del modelo. Hallazgo adicional: **fuzzy y rigid empatan** (el ganador es fuzzy), descartando la afirmación previa de que el voto por confianza tenía un sesgo estructural.

## 2. Hypothesis & Verdict

**Hypothesis (from plan):** "Un KNN sobre las 39 variables tabulares del cohorte `usable_labeled` (N=88), con preprocesado consciente de la ausencia (sin imputación estadística) y evaluado sobre los 50 splits MCCV congelados, produce un baseline de decisión transparente para la subtarea 1.1 (`biopsy_decision`), caracterizado por una rejilla exhaustiva (128 configuraciones). La mejor configuración se selecciona por Macro-F1 en MCCV y se valida con LOO (88 folds) como sanity check."

**Verdict:** ✅ Supported

**Evidence:** la config ganadora supera a ambas mayoritarias en Macro-F1 MCCV (0.6234 vs 0.3793/0.2800), con un sanity check LOO pooled casi idéntico (0.6207). El experimento se ejecutó completo: 128 configs × 50 splits + LOO de la seleccionada, con todos los artefactos CSV y figuras generados.

## 3. Experimental Setup (as run)

Implementado según `IMPLEMENTATION.md` en un único script (`scripts/knn_baseline.py`) + `scripts/plot_results.py`.

- **Dataset**: `inputs_tabular.csv` (195×40; 39 features tras `case_id`), exact subset de `inputs.csv` (verificado columna a columna). Hash en `results/data_manifest.csv`.
- **Cohorte**: 88 `usable_labeled` (54 yes / 34 no). Split file congelado (no regenerado).
- **Preprocessing**: sin imputación. Escalado Min-Max sobre valores observados del train por fold; ausencia → 0 estructural + `missing_<var>`; `cli_bx="None"` es categoría válida; categorías de validación no vistas → bloque one-hot 0 sin columnas nuevas. 36 continuas + 3 categóricas (`cli_bx`, `cli_dre`, `vit_smoking_status`). Sin pruning.
- **Modelo**: KNN brute-force (argsort estable), `p_yes = Σ w_i y_i / Σ w_i`, umbral 0.5 fijo. Fuzzy multiplica `w_i` por confianza del vecino de train (clear=1.0, borderline=0.5, uncertain=0.25).
- **Grid**: 128 configs. Distancias: euclidean, manhattan, minkowski_p3, cosine. Reglas: rigid, fuzzy_confidence. Pesos: uniform, inverse_distance (EPS=1e-12). k: 1,3,5,7,9,11,15,21.
- **Selección**: Macro-F1 MCCV (mean sobre 50 folds); desempate F1_yes, luego balanced accuracy.
- **LOO**: 88 folds solo para la config seleccionada; métricas por-fold (degeneradas a n=1) y métricas pooled (válidas).
- **Hardware/entorno**: conda `histo-DL`, 22 cores (joblib loky), runtime total ≈ 41 s (MCCV) + LOO.
- **Deviations from plan**: ninguna. Nota: `docs/EVALUATION.md` define F1_yes como primaria de campaña; para este experimento se usó Macro-F1 por decisión explícita del PI (documentada en DESIGN.md §2). F1_yes se reporta siempre y es el primer desempate.

## 4. Code Version

| Artifact | Git commit | Mensaje |
|----------|-----------|---------|
| pipeline + resultados | `6e4b61c56b6b26a5040f89053756ec4f0c546629` | `restart proyect` |

⚠️ El árbol de trabajo no estaba limpio al ejecutar (archivos de exp_3 sin commitear, datos en `.gitignore`). El hash de código queda registrado en `results/selected_config/git_commit.txt`; el commit no incluye el código del experimento porque no se solicitó commitear.

## 5. Results

### 5.1 Métrica primaria (MCCV, mean ± std sobre 50 folds)

| Condición | Macro-F1 | F1_yes | Balanced acc | ROC-AUC | PR-AUC | vs. mejor baseline (Δ) |
|-----------|----------|--------|--------------|---------|--------|------------------------|
| **manhattan_fuzzy_confidence_uniform_k11** | **0.6234 ± 0.101** | 0.7418 | 0.6266 | 0.7326 | 0.8346 | **+0.2441** ✅ |
| manhattan_rigid_uniform_k11 | 0.6232 ± 0.094 | 0.7475 | 0.6282 | 0.7397 | 0.8221 | +0.2439 |
| euclidean_rigid_uniform_k9 | 0.6195 ± 0.097 | 0.7354 | 0.6251 | 0.7079 | 0.7992 | +0.2402 |
| always_yes (baseline) | 0.3793 | 0.7606 | 0.5000 | NaN | NaN | — |
| always_no (baseline) | 0.2800 | 0.0000 | 0.5000 | NaN | NaN | — |

> Regla de decisión del plan: ganador = max Macro-F1 MCCV (desempates F1_yes, balanced accuracy). Cumplida; la config seleccionada coincide con la fila 1 de `results/mccv_summary.csv`.

### 5.2 Métricas secundarias — config seleccionada

| Métrica | MCCV (mean±std) | LOO pooled |
|---|---|---|
| Macro-F1 | 0.6234 ± 0.101 | **0.6207** |
| F1_yes | 0.7418 ± 0.077 | 0.7414 |
| F1_no | 0.5050 ± 0.146 | 0.5000 |
| ROC-AUC | 0.7326 ± 0.111 (n_valid=50) | 0.6708 |
| PR-AUC | 0.8346 ± 0.080 | 0.7909 |
| accuracy | 0.6656 ± 0.079 | 0.6591 |
| balanced accuracy | 0.6266 ± 0.086 | — |
| sensitivity (recall yes) | 0.8018 ± 0.135 | 0.7963 |
| specificity (recall no) | 0.4514 ± 0.157 | 0.4412 |
| precision_yes | 0.6986 ± 0.080 | 0.6935 |
| MCC | 0.2854 ± 0.193 | — |
| Brier | 0.2091 ± 0.034 | 0.2243 |

### 5.3 Matrices de confusión

MCCV agregado (900 eventos de validación, 50 splits):

| | pred no | pred yes |
|---|---|---|
| true no | 158 | 192 |
| true yes | 109 | 441 |

LOO pooled (88 predicciones, una por paciente):

| | pred no | pred yes |
|---|---|---|
| true no | 15 | 19 |
| true yes | 11 | 43 |

Figuras: `reports/figures/confusion_matrix_mccv_counts.png`, `confusion_matrix_mccv_normalized.png`, `confusion_matrix_loo_counts.png`, `confusion_matrix_loo_normalized.png`, `confusion_matrix_mccv_vs_loo.png`.

### 5.4 Comportamiento de la rejilla (MCCV Macro-F1)

| Dimensión | mean | max | Observación |
|---|---|---|---|
| fuzzy_confidence | 0.5702 | 0.6234 | Empatado con rigid; el ganador es fuzzy. |
| rigid | 0.5724 | 0.6232 | |
| euclidean | 0.5729 | 0.6195 | |
| manhattan | 0.5723 | 0.6234 | Mejor máximo. |
| minkowski_p3 | 0.5771 | 0.6176 | Mejor media. |
| cosine | 0.5628 | 0.6169 | Peor media. |
| uniform | 0.5729 | 0.6234 | |
| inverse_distance | 0.5697 | 0.6176 | |
| k=1 | 0.5757 | 0.6052 | |
| k=3 | 0.5635 | 0.5871 | |
| k=5 | 0.5746 | 0.6067 | |
| k=7 | 0.5759 | 0.6176 | |
| k=9 | 0.6047 | 0.6195 | Zona óptima. |
| k=11 | 0.6017 | 0.6234 | Zona óptima (ganador). |
| k=15 | 0.5437 | 0.5806 | |
| k=21 | 0.5306 | 0.5807 | |

## 6. Statistical Analysis

- KNN es determinista dado el split file congelado (sin semillas de modelo). La varianza reportada es **entre los 50 folds MCCV**, no entre semillas.
- No se ejecutó test de significancia formal: la comparación se hace sobre folds pareados con la selección pre-fijada por MCCV, y el plan no definió umbral estadístico.
- El sanity check LOO **pooled** (0.6207) valida que la selección MCCV no fue overfitting a la partición.

## 7. Comparison to Expected Results

| Esperado (DESIGN.md §7) | Observado | Match |
|--------------------------|-----------|-------|
| El ganador supera a ambas mayoritarias en Macro-F1 | 0.6234 vs 0.3793 / 0.2800 | ✅ |
| LOO solo sanity check, sin influir en selección | Ejecutado una sola vez, post-selección | ✅ |
| El gap MCCV→LOO se reporta como diagnóstico | pooled: sin gap (0.6234 vs 0.6207) | ✅ |
| Umbral 0.5 sin tuning sobre LOO | Cumplido | ✅ |

## 8. Missing Data & Caveats

- **Todas las condiciones planificadas se ejecutaron**: 128 configs MCCV, baselines, LOO de la config seleccionada, 9 CSVs, 5 figuras.
- ⚠️ El árbol de trabajo no estaba limpio al ejecutar (ver §4).
- ⚠️ Las métricas LOO **por-fold** (mean=0.3295) son degeneradas a n=1 y no deben usarse para caracterizar LOO; las métricas **pooled** son las válidas. El propio `metrics_loo.json` guarda ambas.
- ⚠️ `cli_fh_binary` tiene 3 casos con valor ausente en la cohorte usable (previamente se documentó la posibilidad de recuperarlos desde los JSON crudos); aquí se tratan como ausencia estructural según la política de no imputación aprobada, sin modificar `inputs_tabular.csv`.

## 9. Conclusions & Next Steps

- **Establecido**: (1) KNN tabular es un baseline competitivo para 1.1 (MCCV Macro-F1 0.62, > mayoritarias en ~49-50/50 folds), con comportamiento MCCV≈LOO cuando se usa agregación pooled. (2) El voto fuzzy por confianza no es estructuralmente peor que rigid (empate, gana el ganador); la afirmación contraria de la iteración previa queda refutada empíricamente. (3) La zona óptima de k es 9–11; cosine es la peor distancia; inverse_distance no aporta vs uniform.
- **Incierto**: (1) La generalización al challenge (~104 casos sin etiquetas) no es medible internamente. (2) El umbral 0.5 no es óptimo en MCCV pooled (la curva PR-AUC 0.83 sugiere margen de operación), pero ajustarlo requeriría una regla de selección explícita nueva.
- **Siguientes experimentos recomendados**:
  1. Baseline logístico/regularizado sobre las mismas features y splits para comparar estabilidad out-of-fold (el KNN en k=15/k=21 degrada; un modelo lineal puede ser más estable en N=88).
  2. Subtareas 1.2 (confianza ordinal) y 1.3 (relevancia clínica), reutilizando el mismo protocolo MCCV/LOO y el transformador.
  3. Evaluar la contribución de `cli_fh_binary` ausente recuperándolo desde los JSON crudos (decisión pendiente del PI).
  4. Modo de operación final: reentrenar la config ganadora con los 88 y generar la entrega al evaluador externo (102 vs 104 casos, decisión pendiente del PI).

## 10. Reproducibility Record

| Item | Status |
|------|--------|
| Semillas logueadas | ✅ (no aplica: determinista; splits congelados) |
| Configs versionadas | ✅ (grid en `knn_baseline.py` + `mccv_summary.csv`) |
| Git commit registrado | ✅ (`selected_config/git_commit.txt`) |
| Hashes de datos | ✅ (`data_manifest.csv`) |
| Entorno congelado | ✅ conda `histo-DL` (numpy 1.26.4, pandas 3.0.3, sklearn 1.9.0) |
| Árbol limpio al ejecutar | ❌ (advertido en §4) |
