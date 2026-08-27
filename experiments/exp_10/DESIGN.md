# Experiment Design: Trimodal Hard-KNN vs Fuzzy-KNN Late Fusion for Task 2 Post-Biopsy Treatment Recommendation

**Experiment**: `experiments/exp_10/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)  
**Date**: 2026-08-24  
**Author**: Jonathan Tarquino (PI) & AI Co-Researcher (Deep Learning & Clinical Reasoning Specialist)  
**Status**: Draft — pending PI approval  

---

## 1. Hypothesis

**H1 (Hard-KNN)**: La fusión tardía trimodal hard-KNN — tabular (k=3, uniforme, Euclidiana) + MRI (EmbedKit 1024D, k=3, uniforme, Euclidiana) + texto (TF-IDF + PCA, k=1, uniforme, coseno) — con ponderación soft-voting $\mathbf{w} \in \Delta^2$, alcanza $\overline{\text{Macro-F1}} > 0.50$ en el protocolo MCCV-100 sobre la cohorte trimodal completa de Task 2 ($N=70$).

**H2 (Fuzzy-KNN)**: La fusión tardía trimodal fuzzy-KNN — con *soft targets* guiados por confianza ($c$: `clear`$=1.0$, `borderline`$=0.5$) y extensión multiclase $y_k = \frac14 + \frac34 c$ — alcanza $\overline{\text{Macro-F1}}$ comparable a H1 ($|\Delta| \le 0.03$) con menor Brier score, demostrando que la degradación suave del etiquetado mitiga la escasez de clases minoritarias.

**H3 (Head-to-Head)**: La comparación pareada H1 vs H2 sobre las mismas 70 particiones MCCV-100 determina si la cuantización hard ($\{0,1\}$) o el etiquetado suave por confianza es estadísticamente superior (Wilcoxon pareado, $\alpha = 0.05$) para la recomendación de tratamiento post-biopsia.

---

## 2. Experimental Setup

- **Dataset(s)**:
  - Cohorte oficial CHIMERA Task 2 (`data/raw/train_release/task2/`, 153 carpetas totales).
  - **Cohorte trimodal completa**: $N=70$ con ground truth + MRI 1024D + clinical data + structured prompt.
  - Casos GT excluidos por modalidad faltante: `T2-042`, `T2-044` (sin `MRI image` no vacío).
- **Distribución de clases (N=70)**:

| Clase | Frecuencia | Proporción |
| :--- | :---: | :---: |
| `active_treatment` | 31 | 44.3% |
| `active_surveillance` | 25 | 35.7% |
| `continued_surveillance` | 12 | 17.1% |
| `watchful_waiting` | 2 | 2.9% |

- **Distribución de confianza (N=70)**: `clear` = 56, `borderline` = 14, `uncertain` = 0.
- **Protocolo de Partición**: MCCV-100 (Fase A) con **estratificación de 4 clases** (70% train / 30% val por split), semilla fija `seed=42`, particiones versionadas en `data/processed/task2/mccv_splits_100.csv`.
  - N por split: `train ≈ 49`, `val ≈ 21`.
  - Balanceo por clase garantizado: mín. 1 caso `watchful_waiting` en train.
- **Fase B (LOOCV)**: Evaluación final con hiperparámetros congelados de Fase A sobre $N=70$.

---

## 3. Feature Spaces

### 3.1 Tabular (structured prompt)
- Campos numéricos/categóricos del structured prompt: `age, psa, months, pirads, psad, psav, psap, vol, dre, bx, cspca, isup (bx_isup), gl_prim, gl_sec, ct`, comorbilidades binarias (`medhx`/`pmhx`), `active_treatment_flag`.
- **Auditoría de fuga**: `active_treatment_flag` se excluye del entrenamiento (es el target revelado del caso). Los campos `bx_isup`, `bx_gl_prim`, `bx_gl_sec` se **incluyen** porque son post-biopsia legítimos para Task 2 (el desafío los entrega en el prompt).
- Encoding: numéricos `MinMaxScaler`, categóricos `OneHotEncoder` (fit solo en train split).

### 3.2 MRI (modality-level neural representations)
- Vector 1024D bajo la clave `"MRI image"` de `prostate-modality-level-neural-representations.json`.
- Normalización: `StandardScaler` por split (sin PCA en H1; PCA-64 opcional en red de búsqueda).

### 3.3 Texto (clinical free-text)
- Fuente: `prostate-treatment-decision-clinical-data.json` (campos: `radiology_report`, `previous_notes` concatenados, `psa_trend`, `laboratory_results`, `family_history`, `pathology_report`).
- Vectorización: spaCy `en_core_web_sm` (lematización + filtro stopwords) → TF-IDF. En H1: `max_features=500` + PCA. En H2: vocabulario completo + PCA.
- **Nota crítica**: `pathology_report` contiene biopsia (legítima para Task 2); no se elimina.

---

## 4. Soft-Target Formulation (Fuzzy-KNN, H2)

Para GT clase $k$ con confianza $c \in \{1.0\ (\text{clear}),\ 0.5\ (\text{borderline})\}$:

```
y_k          = 0.25 + 0.75 * c
y_{j ≠ k}    = (1 - y_k) / 3   # distribución uniforme del resto
```

Casos extremos:
- `clear` sobre clase $k$: $\mathbf{y} = [0, 0, 1, 0]$ (one-hot perfecto).
- `borderline` sobre clase $k$: $\mathbf{y}_k = 0.625$, $\mathbf{y}_{j \ne k} = 0.125$ (masa probabilística distribuida).

El fuzzy-KNN (implementado vía `KNeighborsRegressor`) predice directamente los 4 soft targets; la clase final se obtiene con $\arg\max$.

---

## 5. Conditions

| Condición | Modalidad | Modelo | Hiperparámetros (grid Fase A) |
| :--- | :--- | :--- | :--- |
| $C_1$ | Tabular | `KNeighborsClassifier` (hard) | $k \in \{1, 3, 5, 7\}$, weights `{uniform, distance}`, metric `{euclidean, manhattan}` |
| $C_2$ | MRI | `KNeighborsClassifier` (hard) | $k \in \{1, 3, 5\}$, weights `{uniform, distance}`, metric `{euclidean}` |
| $C_3$ | Texto | `KNeighborsClassifier` (hard) | $k \in \{1, 3, 5\}$, weights `{uniform}`, metric `{cosine}` |
| $C_4$ | Tabular | `KNeighborsRegressor` (fuzzy) | $k \in \{1, 3, 5\}$, weights `{uniform, distance}` |
| $C_5$ | MRI | `KNeighborsRegressor` (fuzzy) | $k \in \{1, 3, 5\}$, weights `{uniform, distance}` |
| $C_6$ | Texto | `KNeighborsRegressor` (fuzzy) | $k \in \{1, 3, 5\}$, weights `{uniform}`, metric `{cosine}` |
| $C_7^{H1}$ | **Fusión H1** | Soft-voting sobre $C_1^*:C_3^*$ | $\mathbf{w} \in \Delta^2$ (grid $0.05$, $\lvert \mathcal{W} \rvert = 231$ combinaciones) |
| $C_8^{H2}$ | **Fusión H2** | Soft-voting sobre $C_4^*:C_6^*$ | $\mathbf{w} \in \Delta^2$ (grid $0.05$, $\lvert \mathcal{W} \rvert = 231$ combinaciones) |

**Fusión (H1/H2)**: promediado ponderado de probabilidades (o soft targets) por modalidad:
$$\hat{p}_j = \sum_{m \in \{tab, mri, txt\}} w_m \cdot \hat{p}_j^{(m)}, \quad \mathbf{w} \in \Delta^2$$

Decisión final: $\hat{y} = \arg\max_j \hat{p}_j$ (sin umbral binarizable en 4 clases).

---

## 6. Evaluation Protocol

- **Métricas primarias**:
  1. $\overline{\text{Macro-F1}}_{\text{MCCV-100}}$ (media sobre 100 splits).
  2. **Brier score multiclase** (soft-target rollout): $\frac{1}{N}\sum_i \sum_j (p_{ij} - y_{ij})^2$.
- **Métricas secundarias**:
  3. **Micro-F1** (agregado global).
  4. **Weighted-F1** (ponderado por soporte de clase).
  5. Macro-AUROC (One-vs-Rest).
  6. Per-class F1 (crítico para `watchful_waiting`, $n=2$).
- **Pruebas de hipótesis**:
  - Wilcoxon pareado (two-sided, $\alpha=0.05$) H1 vs H2 sobre los 100 splits (Macro-F1 y Brier).
  - McNemar exacto sobre matriz 4×4 de decisiones pareadas (opcional si la evidencia es marginal).
- **Gobernanza**: métricas calculadas exclusivamente vía `chimera_metrics` (o `sklearn.metrics` homologado para 4 clases), nunca ad-hoc.

---

## 7. Storage Footprint (Minimal)

Tras ejecución, se persisten **únicamente**:

```
experiments/exp_10/
├── DESIGN.md                  <- this file
├── IMPLEMENTATION.md          <- plan de implementación (aprobación previa)
├── results/
│   ├── version.txt            <- git commit + timestamp
│   ├── aggregate_metrics.json <- medias ± std por condición (Macro-F1, Micro-F1, Weighted-F1, AUROC, Brier, per-class F1)
│   ├── split_metrics.json     <- métricas por split (100 filas) — compacto ~20 KB
│   ├── wilcoxon_tests.json    <- comparaciones pareadas H1 vs H2
│   └── best_hyperparams.json  <- config óptima por condición (Fase A)
└── reports/
    └── summary.md             <- informe final
```

**Restricción explícita**: NO se guardan CSVs de predicciones OOF completas (70k filas × 8 condiciones = ~10 MB). Solo JSON compactos de métricas agregadas y por split. Los arrays de probabilidades solo residen en memoria durante la ejecución.

---

## 8. Runtime Mitigations

- **Poda de Fase A**: grids unimodales reducidos (máx. ~16 configs tab, ~12 MRI, ~6 texto) → ~34 × 100 splits = 3,400 evaluaciones KNN (inferiores a los 24,120 de `exp_4`).
- **Fusión**: 231 combinaciones × 100 splits × 2 familias = 46,200 soft-votings sobre arrays 70×4 (trivial en memoria).
- **MRI**: PCA-64 pre-calculado por split (no por condición) para eliminar 16× de cómputo KNN.
- **Estimación**: < 15 minutos en CPU local (16 hilos), ejecución en `tmux:0`.

---

## 9. Risks & Mitigations

| Riesgo | Prob./Sev. | Mitigación |
| :--- | :---: | :--- |
| **Colapso de `watchful_waiting` (n=2)** | Alta / Alta | El fuzzy-KNN distribuye masa de `borderline` evitando $\arg\max$ arbitrario; se reporta per-class F1 con intervalo y se pre-registra que la clase no admite inferencia robusta (limita Macro-F1 máxima alcanzable). |
| **Fuga de `active_treatment_flag`** | Media / Crítica | Auditoría estática antes de entrenar: el campo se elimina del frame tabular. |
| **Correlación intermodal** (MRI 1024D con $N=70$) | Media / Media | PCC/KNN no estima covarianza; sin riesgo de singularidad. Impacto solo en varianza del estimador — mitigado con MCCV-100. |
| **Desbalance severo en splits** | Alta / Media | Estratificación por clase con garantía de $\ge 1$ caso `watchful_waiting` en train y en val (si es factible); si no, split se descarta y se re-muestrea (rejection sampling determinista). |
| **Overfitting en Fase A (selección de pesos)** | Media / Media | Selección de $\mathbf{w}^*$ por mayor Macro-F1 medio en val; segunda fase LOOCV congelada reporta el rendimiento honesto. |

---

## 10. Baseline(s)

| Baseline | Descripción | Métrica esperada (Macro-F1) |
| :--- | :--- | :---: |
| $B_0$ | **Mayoritario**: predice siempre `active_treatment` | $\approx 0.11$ (clase mayoritaria 44.3%) |
| $B_1$ | **Unimodal tabular hard-KNN** ($C_1$ mejor single) | Referencia exp_8 Task 1 análoga ≈ 0.63 (Task 2 cae por 4 clases) |
| $B_2$ | **Unimodal texto hard-KNN** ($C_3$ mejor single) | Referencia exp_8 Task 1 ≈ 0.70 (ajustado a 4 clases) |
| $B_3^{H1}$ | Fusión H1 (hard) | Hipótesis > 0.50 |
| $B_4^{H2}$ | Fusión H2 (fuzzy) | Hipótesis comparable a $B_3$ con Brier menor |

---

## 11. Expected Results & Decision Rules

- **Éxito primario (H1)**: $\overline{\text{Macro-F1}}_{H1} > 0.50$ con $p < 0.001$ vs $B_0$ (Wilcoxon pareado).
- **Éxito primario (H2)**: $\overline{\text{Macro-F1}}_{H2} \ge \overline{\text{Macro-F1}}_{H1} - 0.03$ **y** Brier$_{H2}$ < Brier$_{H1}$.
- **Criterio de superioridad (H3)**: H2 es superior si Wilcoxon pareado $p < 0.05$ en Macro-F1 **o** en Brier, y nunca significativamente peor en la otra métrica.
- **Fracaso/ajuste**: si ambas fusiones $\le 0.40$, se revisa la ingeniería de features (p.ej., añadir `ipss`, `vitals`, `psa_trend` como curva temporal) y se propone `exp_11`.

---

## 12. Reproducibility Checklist

- [ ] Particiones MCCV-100 versionadas en `data/processed/task2/mccv_splits_100.csv`
- [ ] Semillas fijas (`seed=42`) para splits y KNN deterministas
- [ ] Auditoría de fuga (`active_treatment_flag` excluido) registrada en `results/version.txt`
- [ ] Configs de hiperparámetros registradas en `results/best_hyperparams.json`
- [ ] Cálculo de métricas exclusivamente vía `chimera_metrics` / `sklearn.metrics` homologado
- [ ] Almacenamiento mínimo (sin CSVs de predicciones completas)
- [ ] Ejecución en `tmux:0` con reportería compacta en consola

---

## 13. Next Steps

1. **Aprobación formal del PI** sobre este `DESIGN.md`.
2. Generar `IMPLEMENTATION.md` con scripts:
   - `scripts/preprocess_task2.py` → CSVs canónicos (tab, mri, text, gt) en `data/processed/task2/`.
   - `scripts/generate_mccv_splits.py` → 100 splits estratificados a 4 clases.
   - `scripts/train.py` → Fase A (grids unimodales + fusión), Fase B (LOOCV congelada).
3. Ejecutar benchmark en `tmux:0` y persistir `results/`.
4. Invocar `experiment-reporter` → `reports/summary.md`.
5. Actualizar `.logbook.md` y `.discussion.md` tras el consenso.

---