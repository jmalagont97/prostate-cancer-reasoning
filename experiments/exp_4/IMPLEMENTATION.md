# Implementation Plan: Feature-Subset Ablation (exp_4)

**Experiment**: experiments/exp_4/ · **Project**: pathology-reasoning · **Date**: 2026-08-15 · **Status**: Approved

---

## 1. Overview

Reutiliza la infraestructura de `exp_3` (mismo pipeline, mismos splits, misma rejilla de 128 configs) y aplica **una sola modificación controlada**: el filtro de features por >50% de ausencia, fijado globalmente antes del MCCV. Tres scripts:

- `scripts/knn_baseline.py` — pipeline completo (idéntico a `exp_3` salvo el filtro + artefactos nuevos de selección).
- `scripts/plot_results.py` — figuras de matrices de confusión de la config seleccionada (adaptado de `exp_3`).
- `scripts/compare_experiments.py` — comparación pareada exp_3 (39 vars) vs exp_4 (37 vars) → `comparison_39_vs_37.csv` + figura.

Ejecución en conda env `histo-DL`:

```bash
python experiments/exp_4/scripts/knn_baseline.py
python experiments/exp_4/scripts/plot_results.py
python experiments/exp_4/scripts/compare_experiments.py
```

## 2. Module Layout

```
experiments/exp_4/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   ├── __init__.py
│   ├── knn_baseline.py
│   ├── plot_results.py
│   └── compare_experiments.py
├── results/
│   ├── feature_missingness.csv
│   ├── feature_selection_manifest.csv
│   ├── comparison_39_vs_37.csv
│   ├── data_manifest.csv
│   ├── mccv_summary.csv
│   ├── mccv_fold_metrics.csv
│   ├── mccv_oof_predictions.csv
│   ├── loo_predictions.csv
│   ├── confusion_matrices_mccv.csv
│   ├── confusion_matrix_loo.csv
│   ├── classification_report.csv
│   ├── baseline_metrics.csv
│   └── selected_config/
│       ├── hyperparameters.json
│       ├── metrics_mccv.json
│       ├── metrics_loo.json
│       └── git_commit.txt
└── reports/
    ├── summary.md          (after execution)
    └── figures/
        ├── confusion_matrix_mccv_counts.png
        ├── confusion_matrix_mccv_normalized.png
        ├── confusion_matrix_loo_counts.png
        ├── confusion_matrix_loo_normalized.png
        ├── confusion_matrix_mccv_vs_loo.png
        └── comparison_macro_f1_39_vs_37.png
```

## 3. knn_baseline.py — cambios vs exp_3

### 3.1 Constantes y filtro
- `EXP_DIR = PROJECT_ROOT / "experiments" / "exp_4"`.
- `ESSENTIAL_FEATURES` (10): `cli_age, cli_fh_binary, cli_cspca, cli_pirads, cli_vol, cli_psa, cli_comorbidity_count, cli_psad, cli_dre, cli_bx`.
- `MISSINGNESS_THRESHOLD = 0.5` (estricto: retener si `missing_rate <= 0.5` o esencial).
- En `main()` (antes de entrenar): calcular `missing_rate` por variable sobre los 88 casos (string vacío) y fijar:
  - `REMOVED = [c for c in ALL_FEATURES if missing_rate(c) > 0.5 and c not in ESSENTIAL_FEATURES]` → esperado `{path_hist_bx_gl_tert, lab_hemoglobin_g_dl}`.
  - `RETAINED = [c for c in ALL_FEATURES if c not in REMOVED]` → 37.
  - `CONTINUOUS_USED` = `[c in CONTINUOUS_COLS and retained]` (esperado 34); `CATEGORICAL_USED` = ídem (esperado 3). El transformador y `ALL_FEATURES_USED` (para indicadores `missing_`) usan solo variables retenidas.
- Assertions: `len(RETAINED) == 37`, `len(REMOVED) == 2`, esenciales ⊆ RETAINED, `missing_rate(removed) > 0.5` para cada `REMOVED`.
- Grid, distancias, pesos, k, confianza, métricas, MCCV/LOO: **idénticos** a `exp_3`.

### 3.2 Artefactos nuevos
- `feature_missingness.csv`: `variable, n_missing, missing_rate, is_essential, retained`.
- `feature_selection_manifest.csv`: `rule, threshold, cohort, n_cases, n_features_total, n_retained, n_removed, removed_features, retained_features, essential_features, fixed_at`.
- `data_manifest.csv`: incluye `n_features=37`, `pruning=missingness_gt50`, `removed_features`, `missingness_threshold=0.5`.
- Guarda `git_commit.txt` y el resto de CSVs como en `exp_3`.

## 4. plot_results.py — cambios vs exp_3

Solo cambia `EXP_DIR` → `exp_4`. Mismas 5 figuras de matrices de confusión (config seleccionada). Textos idénticos (MCCV = 900 eventos; LOO = 88 pacientes).

## 5. compare_experiments.py

Lee `exp_3/results/mccv_fold_metrics.csv` y `exp_4/results/mccv_fold_metrics.csv` (ambos 6400 filas, mismos `config_id`).

Por cada config (128):
- `Macro_F1_mean_39`, `Macro_F1_mean_37`, `delta_mean = mean_37 - mean_39`.
- Delta pareado por fold (`Macro_F1_37[split] - Macro_F1_39[split]`, 50 valores), `delta_std`, `delta_ci95` (t de Student), `wilcoxon_p` (scipy `wilcoxon`).
- Filas marcadas: `exp3_winner=manhattan_fuzzy_confidence_uniform_k11`, `exp4_winner=<config_id>` (leído de `exp_4/results/selected_config/hyperparameters.json`).

Outputs:
- `results/comparison_39_vs_37.csv`.
- `reports/figures/comparison_macro_f1_39_vs_37.png`: scatter 128 configs (x=mean39, y=mean37) + línea identidad + anotaciones de los dos ganadores.

## 6. Verification after execution

- `feature_selection_manifest.csv`: removed == `path_hist_bx_gl_tert,lab_hemoglobin_g_dl`; retained == 37.
- Row counts: `mccv_summary.csv`=128; `mccv_fold_metrics.csv`=6400; `mccv_oof_predictions.csv`=115200; `loo_predictions.csv`=88; `confusion_matrices_mccv.csv`=25600.
- `comparison_39_vs_37.csv`=128 filas; `delta_mean` coincide con diferencia de medias; p-valores finitos.
- Transform sin NaN, finitos (asserts).
- Selected config en rango 1 de `mccv_summary.csv`.
- Figuras existen y no están vacías.
- Hashes de datos coinciden con `exp_3` (`data_manifest.csv`).
