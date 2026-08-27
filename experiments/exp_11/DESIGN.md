# Experiment Design: Trimodal Linear-SVM Late Fusion (Hard SVC + Soft-SVR) for Task 2 Treatment Recommendation

**Experiment**: `experiments/exp_11/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)  
**Date**: 2026-08-25  
**Author**: Jonathan Tarquino (PI) & AI Co-Researcher (Deep Learning & Clinical Reasoning Specialist)  
**Status**: Draft — pending PI approval  
**Based on**: `exp_10` (established H2 fuzzy-KNN as champion, Macro-F1 = 0.5971 ± 0.1258)

---

## 1. Hypothesis

**H1**: La sustitución del estimador base KNN por **SVM con kernel lineal** (SVC con probabilidades Platt) en las tres modalidades mantiene o mejora el rendimiento de la fusión hard: $\overline{\text{Macro-F1}}_{SVM\text{-hard}} \ge \overline{\text{Macro-F1}}_{KNN\text{-hard}} = 0.5863$ (Wilcoxon pareado, $\alpha=0.05$).

**H2**: La familia **fuzzy-SVM** implementada como **Soft-SVR por clase** (regresión sobre los mismos soft-targets 4-d de `exp_10`) alcanza $\overline{\text{Macro-F1}}_{SVM\text{-fuzzy}}$ comparable a la familia fuzzy-KNN ($0.5971$) y conserva la ventaja en calibración (Brier menor) observada en `exp_10`.

**H3 (Cross-Experiment)**: El mejor SVM (hard o soft) **empata o supera** al champion fuzzy-KNN de `exp_10` (Macro-F1 = 0.5971, Brier = 0.5176) sobre las **mismas 100 particiones**, permitiendo atribuir la diferencia exclusivamente al cambio de modelo.

---

## 2. Experimental Setup

- **Dataset**: Cohorte trimodal completa de Task 2 ($N=70$, `data/processed/task2/`) — **idéntica a `exp_10`**.
- **Protocolo de Partición**: **REUTILIZACIÓN EXACTA** de `data/processed/task2/mccv_splits_100.csv` (100 splits estratificados 4-clases, seed=42). NO se regeneran splits — garantiza comparación pareada legítima con `exp_10`.
- **Fase B**: LOOCV congelado sobre $N=70$ con hiperparámetros de Fase A.
- **Preprocesamiento (con cambios respecto a `exp_10`)**:
  - **Tabular**: `StandardScaler` (reemplaza `MinMaxScaler` de exp_10) — crítico para SVM por sensibilidad a la escala.
  - **MRI (1024D)**: `StandardScaler` fitteado en train fold (nuevo — exp_10 no escalaba MRI). Sin PCA.
  - **Texto**: TF-IDF (`H1: max_features=500`, `H2: max_features=None`) — sin cambio, adecuado para SVM lineal.
  - **Leakage**: `active_treatment_flag` excluido; escaladores fit solo en train fold por split.

---

## 3. Model Families

### 3.1 Hard-SVM ($C_1^*$–$C_3^*$)
- `LinearSVC` o `SVC(kernel='linear', probability=True)` por modalidad.
- `probability=True` para obtener probabilidades Platt (continuas, calibradas) — https para fusión.
- **Grid Fase A** (por modalidad):
  - `C ∈ {0.1, 1.0, 10.0}`
  - `class_weight ∈ {None, 'balanced'}`  (abla el desbalance de `watchful_waiting`)

### 3.2 Fuzzy-SVM / Soft-SVR ($C_4^*$–$C_6^*$)
- **`MultiOutputRegressor(LinearSVR())`** sobre las **mismas soft-targets 4-d** de `exp_10` ($y_k = 0.25 + 0.75c$, $y_{j\ne k} = (1-y_k)/3$).
- Un `LinearSVR` por clase (ovo en 4 salidas independientes).
- **Grid Fase A**: `C ∈ {0.1, 1.0, 10.0}`, `epsilon ∈ {0.05, 0.1}`.
- Normalización L1 de las 4 salidas a simplex antes de la fusión (los SVR no garantizan suma 1).

### 3.3 Fusión tardía
- **Soft-voting ponderado** sobre las 4-d probabilidades/salidas normalizadas de las 3 modalidades.
- Simplex $\mathbf{w} \in \Delta^2$ con paso 0.05 (231 combinaciones), maximizando Macro-F1 medio de validación.

---

## 4. Conditions

| Condición | Familia | Modalidad | Modelo | Grid |
| :--- | :--- | :--- | :--- | :--- |
| $C_1$ | Hard | Tabular | `SVC(linear, probability=True)` | C × class_weight |
| $C_2$ | Hard | MRI | `SVC(linear, probability=True)` | C × class_weight |
| $C_3$ | Hard | Texto | `SVC(linear, probability=True)` | C × class_weight |
| $C_4$ | Fuzzy | Tabular | `MultiOutputRegressor(LinearSVR)` | C × epsilon |
| $C_5$ | Fuzzy | MRI | `MultiOutputRegressor(LinearSVR)` | C × epsilon |
| $C_6$ | Fuzzy | Texto | `MultiOutputRegressor(LinearSVR)` | C × epsilon |
| $C_7^{H1}$ | **Fusión Hard-SVM** | Trimodal | Soft-voting sobre $C_1^*$–$C_3^*$ | w ∈ Δ² |
| $C_8^{H2}$ | **Fusión Fuzzy-SVR** | Trimodal | Soft-voting sobre $C_4^*$–$C_6^*$ | w ∈ Δ² |

---

## 5. Evaluation Protocol

**Métricas idénticas a `exp_10`** (comparabilidad directa):
- **Primarias**: $\overline{\text{Macro-F1}}_{\text{MCCV-100}}$, Brier multiclase.
- **Secundarias**: Micro-F1, Weighted-F1, Macro-AUROC (OVR), per-class F1, gate accuracy (`chimera_metrics.task2.compute_treatment_decision_gate`).

**Pruebas de hipótesis**:
1. Wilcoxon pareado (two-sided, $\alpha=0.05$) H1 vs H2 internas (100 splits).
2. **Cross-experiment**: Wilcoxon pareado `exp_11 (H1 y H2)` vs `exp_10 (H1 y H2)` sobre los 100 splits — **particiones compartidas permiten test directo**.
3. McNemar exact sobre matríz 4×4 de decisiones pareadas (si Wilcoxon marginal, $p \in [0.05, 0.10]$).

---

## 6. Baselines

| Baseline | Origen | Referencia |
| :--- | :--- | :--- |
| $B_0$ | Mayoritario (`active_treatment`) | Macro-F1 ≈ 0.11 |
| $B_1$ | `exp_10` H1 (hard-KNN fusion) | Macro-F1 = 0.5863, Brier = 0.5306 |
| $B_2$ | `exp_10` H2 (fuzzy-KNN fusion) | Macro-F1 = 0.5971, Brier = 0.5176 |

---

## 7. Storage Footprint (Minimal)

Idéntico a `exp_10` — solo JSON compactos:

```
experiments/exp_11/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   ├── run_exp11_mccv.py
├── results/
│   ├── version.txt
│   ├── aggregate_metrics.json
│   ├── split_metrics.json
│   ├── best_hyperparams.json
│   └── wilcoxon_tests.json
└── reports/
    └── summary.md
```

SIN CSVs de predicciones OOF. Arrays de probabilidades solo en memoria.

---

## 8. Runtime Mitigations

- **Grid reducido**: 6 configs tabular/MRI/texto × 100 splits = 600 evaluaciones por familia (vs 3,100 en exp_10). Mucho más rápido que KNN (no hay `pairwise_distances`).
- Fusión: 231 × 100 × 2 = 46,200 soft-votings sobre arrays 4-d (trivial).
- Estimación: < 5 min en CPU local (SVM lineal es rápido en N=49).

---

## 9. Risks & Mitigations

| Riesgo | Prob./Sev. | Mitigación |
| :--- | :---: | :--- |
| **SVC no converge en N≈49 con 1024D (MRI)** | Media / Media | `LinearSVC` con `max_iter=5000`, `dual=True` (n_features > n_samples); si no converge → PCA-64 como fallback (pre-registrado). |
| **Probabilidades Platt pobremente calibradas con N pequeña** | Media / Media | `SVC(probability=True)` usa 5-fold CV interna para Platt — documentar; si Brier degrada, usar `LinearSVC.decision_function` → softmax empírico. |
| **Soft-SVR produce salidas >1 o negativas** | Alta / Baja | Normalización L1 + clip a [0,1] antes de fusión. |
| **Overfitting en selección de C** | Media / Media | Selección por mayor Macro-F1 medio en val (100 splits); segunda fase LOOCV congelada honesta. |
| **Class imbalance (`watchful_waiting`)** | Alta / Alta | `class_weight='balanced'` en SVC; per-class F1 con advertencia explícita de n=2. |

---

## 10. Expected Results & Decision Rules

- **Éxito primario (H1)**: $\text{Macro-F1}_{SVM\text{-hard}} \ge 0.5863$ (no significativamente peor, Wilcoxon $p \ge 0.05$)
- **Éxito primario (H2)**: $\text{Macro-F1}_{SVM\text{-fuzzy}} \ge 0.5971$ **o** $\Delta \le -0.02$ con Brier inferior en ≥ 0.01
- **Supremacía (H3)**: si algún SVM supera al fuzzy-KNN champion de forma significativa ($p < 0.05$) → nuevo champion del modelo para Task 2
- **Fracaso/ajuste**: si ambos SVM < 0.55, el problema no es de modelo sino de features — se propone `exp_12` centrado en engineering

---

## 11. Reproducibility Checklist

- [ ] REUTILIZA `data/processed/task2/mccv_splits_100.csv` (comparación pareada válida)
- [ ] `StandardScaler` para tabular y MRI (fit solo en train fold)
- [ ] `active_treatment_flag` excluido (auditoría)
- [ ] Configs registradas en `best_hyperparams.json`
- [ ] Métricas vía `chimera_metrics` + homologación sklearn
- [ ] Almacenamiento mínimo (JSON compactos)

---

## 12. Next Steps

1. **Aprobación formal del PI** sobre este `DESIGN.md`.
2. Generar `IMPLEMENTATION.md` (adaptación del runner de exp_10).
3. Ejecutar en background y persistir `results/`.
4. `experiment-reporter` → `reports/summary.md`.
5. Actualizar `.logbook.md` y `.discussion.md` tras el consenso.

---