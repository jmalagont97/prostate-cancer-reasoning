# Experiment Design: Feature-Subset Ablation (>50% Missingness) on Tabular KNN (exp_4)

**Experiment**: experiments/exp_4/ · **Project**: pathology-reasoning · **Date**: 2026-08-15 · **Status**: Complete

---

## 1. Hypothesis

Eliminar las variables tabulares con más del 50% de ausencia en el cohorte `usable_labeled` (N=88), conservando siempre las 10 variables clínicas esenciales, mejora o mantiene el **Macro-F1** del KNN frente al baseline de `exp_3` (39 variables), al reducir dimensiones con información estructuralmente escasa sin sacrificar las variables clínicas.

## 2. Experimental Setup

- **Feature matrix**: `data/chimera26/preprocessed/task1/inputs_tabular.csv` (195 × 40; 39 variables). Hash en `results/data_manifest.csv`.
- **Targets**: `ground_truth.csv` — `target_biopsy_decision` ∈ {`yes`, `no`}; `target_confidence_code` (para voto fuzzy).
- **Cohorte**: 88 `usable_labeled` (54 yes / 34 no).
- **Filtro de features (regla pre-registrada)**:
  - `missing_rate(c)` = proporción de string vacío en el cohorte usable_labeled (88 casos).
  - Retener `c` si `missing_rate(c) <= 0.5` **o** `c ∈ ESSENTIAL_FEATURES`.
  - La máscara se fija **globalmente, antes de cualquier split MCCV** (sin usar `y` ni casos de challenge), y se congela para todos los folds.
  - **Umbral estricto `>0.5`** (una variable con 50.0% exacto se conserva).
- **Variables eliminadas (2)**: `path_hist_bx_gl_tert` (87/88 = 98.86%), `lab_hemoglobin_g_dl` (65/88 = 73.86%). Ambas superan 50% en los 50 folds de train (98.6–100% y 70–80%), por lo que un filtro per-fold produciría el mismo conjunto.
- **Variables retenidas (37)**: 34 continuas + 3 categóricas (`cli_bx`, `cli_dre`, `vit_smoking_status`).
- **Variables esenciales (10, protegidas siempre)**: `cli_age`, `cli_fh_binary`, `cli_cspca`, `cli_pirads`, `cli_vol`, `cli_psa`, `cli_comorbidity_count`, `cli_psad`, `cli_dre`, `cli_bx`. Ninguna supera 50% en estos datos; la cláusula es defensiva y queda registrada como regla permanente.
- **Splits (congelados)**: `mccv_loocv_splits.csv`. MCCV = 50 splits de 70 train / 18 val; LOO = 88 folds.
- **Métrica de selección**: **Macro-F1** en MCCV (misma desviación explícita que `exp_3` respecto a `docs/EVALUATION.md` §3.1, aprobada por el PI). Desempates: F1_yes, luego balanced accuracy.
- **Umbral de decisión**: fijo en 0.5 para todas las configs. Sin tuning sobre LOO.
- **LOO**: solo para la config seleccionada, como sanity check; no influye en selección.

## 3. Missingness & Preprocessing (sin imputación)

Idéntico a `exp_3`:

- Lectura como strings (`keep_default_na=False, na_values=[], dtype=str`); ausencia = string vacío.
- **Continuas**: Min-Max sobre valores observados del train por fold. Ausencia → 0 estructural; observado → `clip((x-min)/(max-min),0,1)`; constante → 1.0. Indicador `missing_<var>` por variable **retenida**.
- **Categóricas**: one-hot con categorías observadas en train. `cli_bx="None"` es categoría válida; `cli_dre="Not done"` es categoría válida. Ausencia → bloque 0 + `missing_<var>=1`.
- Transformadores ajustados por fold de train y aplicados a validación. Sin estadísticas globales.

## 4. Grid: 128 Configurations

| Dimensión | Valores |
|---|---|
| distance | euclidean, manhattan, minkowski_p3, cosine |
| rule | rigid, fuzzy_confidence |
| weight | uniform, inverse_distance |
| k | 1, 3, 5, 7, 9, 11, 15, 21 |

- Rigid: `w_i = distance_weight_i`; Fuzzy: `w_i = confidence_weight_i × distance_weight_i` (clear=1.0, borderline=0.5, uncertain=0.25), train-side.
- `p_yes = Σ w_i y_i / Σ w_i`; `y_pred = 1[p_yes ≥ 0.5]`.
- `inverse_distance`: `w = 1/max(d, 1e-12)`; argsort estable; cosine guard anti-norma-cero.

## 5. Evaluation Protocol

Per fold (MCCV val): accuracy, balanced accuracy, sensitivity, specificity, precision_yes, F1_yes, F1_no, Macro-F1, MCC, Brier, ROC-AUC, PR-AUC (NaN si fold monoclase).

Per config (agregado 50 folds): mean/std/min/max/n_valid.

**LOO (config seleccionada)**: métricas por-fold (n=1) + pooled (88) ROC-AUC/PR-AUC; matriz de confusión pooled 2×2.

**Baselines**: `always_yes`, `always_no` (MCCV + LOO pooled).

## 6. Outputs

### CSVs (results/) — mismos que exp_3, más artefactos del filtro y comparación
- `feature_missingness.csv` — por variable: `n_missing`, `missing_rate` en los 88 casos, `is_essential`, `retained`.
- `feature_selection_manifest.csv` — regla, umbral, listas retenidas/eliminadas, cohorte y fecha de fijación.
- `mccv_summary.csv` (128), `mccv_fold_metrics.csv` (6400), `mccv_oof_predictions.csv` (115200), `loo_predictions.csv` (88), `confusion_matrices_mccv.csv` (25600), `confusion_matrix_loo.csv`, `classification_report.csv`, `baseline_metrics.csv`, `data_manifest.csv`.
- `comparison_39_vs_37.csv` — comparación pareada por config (128 filas): mean Macro-F1 (39 vars) vs (37 vars), delta, p-valor Wilcoxon pareado (50 folds), y fila del ganador de exp_4 con su equivalente en exp_3.

### Selected config (results/selected_config/)
- `hyperparameters.json`, `metrics_mccv.json`, `metrics_loo.json`, `git_commit.txt`.

### Figuras (reports/figures/)
- `confusion_matrix_mccv_counts.png`, `confusion_matrix_mccv_normalized.png`, `confusion_matrix_loo_counts.png`, `confusion_matrix_loo_normalized.png`, `confusion_matrix_mccv_vs_loo.png` (config seleccionada).
- `comparison_macro_f1_39_vs_37.png` — scatter de las 128 configs (x=Macro-F1 39 vars, y=Macro-F1 37 vars) con línea identidad; se marcan los ganadores de ambos experimentos.

## 7. Decision Rules

1. Rango de las 128 configs por MCCV mean Macro-F1 (desempates F1_yes, balanced accuracy). El ganador es el baseline de `exp_4`; LOO solo para él.
2. **Comparación vs exp_3** (exploratoria, config fija): para cada config, delta = Macro-F1_37 − Macro-F1_39 a nivel fold (pareado, 50 folds) y test Wilcoxon signed-rank; interpretación:
   - Mejora: delta medio positivo y p < 0.05.
   - Sin diferencia concluyente: p ≥ 0.05 o intervalo pareado cruza 0.
   - Degradación: delta negativo consistente (p < 0.05).
3. La comparación ganador-vs-ganador se reporta por separado (no como prueba formal, porque la selección dentro de los mismos 50 splits induce dependencia; se señala explícitamente).
4. Si el ganador de exp_4 supera a ambas mayoritarias en Macro-F1, el baseline filtrado se declara adecuado para anclaje de campaña.

## 8. Risks & Mitigations

- **Selección post-hoc**: la regla (>50%, global, sobre 88 casos sin usar `y`) se pre-registra en este DESIGN.md antes de ejecutar; la comparación contra exp_3 se declara exploratoria.
- **Leakage de selección**: el filtro no usa etiquetas; la máscara global usa la estructura de ausencia del cohorte completo (incluye los 18 val de cada split). Mitigación: regla fijada antes de correr + verificación de que las 2 variables eliminadas superan 50% en 50/50 folds de train (hecho). Sin variables cerca del umbral (98.9% / 73.9% vs 50%).
- **Desigualdad de comparación**: exp_3 y exp_4 comparten splits congelados y código idéntico salvo el filtro → comparación pareada válida por config.
- **Coherencia clínica**: se elimina `path_hist_bx_gl_tert` pero se conservan `path_hist_bx_isup/prim/sec` (26.1% cada una); la decisión es puramente por umbral y así queda registrado.
- **Árbol no limpio**: registrado en `git_commit.txt`.

## 9. Reproducibility Checklist

- [x] Split file congelado (nunca regenerado), hash registrado.
- [x] Regla del filtro pre-registrada aquí (§2) con umbral estricto `>0.5`.
- [x] Máscara global fijada y exportada a `feature_selection_manifest.csv`.
- [x] Cohorte = 88 usable_labeled; balance 54/34.
- [x] Sin semillas (KNN determinista + splits congelados).
- [x] Preprocesado por fold de train.
- [x] Sin imputación.
- [x] Entorno conda `histo-DL` (numpy 1.26.4, pandas 3.0.3, scikit-learn 1.9.0, matplotlib 3.10.9).
- [x] Hashes de datos en `results/data_manifest.csv`.
- [x] Git commit hash en `results/selected_config/git_commit.txt`.

## 10. Next Steps

1. Plan aprobado por el PI en conversación (2026-08-15); este DESIGN.md lo fija por escrito.
2. Implementación en `scripts/` (ver `IMPLEMENTATION.md`).
3. Ejecutar, verificar y reportar en `reports/summary.md`.
