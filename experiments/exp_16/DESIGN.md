# Experiment Design: Replicación de Fusión Trimodal Fuzzy-SVR para Decisión Terapéutica (Subtarea 2.1) bajo MCCV-30
**Experiment**: `experiments/exp_16/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)  
**Date**: 2026-08-30  
**Author**: Juan Sebastian Malagón Torres  
**Status**: Draft  

---

## 1. Hypothesis

La replicación de la arquitectura champion de `exp_11` (fusión tardía trimodal con `MultiOutputRegressor(LinearSVR)` entrenado sobre soft-targets ponderados por confianza clínica) sobre el protocolo estandarizado MCCV-30 (`data/processed/task2/mccv_splits_30.csv`) mantendrá una calibración superior (Brier $\le 0.40$) y un Macro-F1 competitivo ($\ge 0.60$) frente al baseline Hard-SVC, evaluado exclusivamente a través del paquete oficial `chimera_metrics`.

## 2. Experimental Setup

- **Dataset**: Cohorte canónica de Task 2 ($N=70$ casos estructurados en `data/processed/task2/`):
  - `data/processed/task2/clinical_tabular_task2.csv` ($70 \times 33$)
  - `data/processed/task2/mri_embeddings_task2.csv` ($70 \times 1025$)
  - `data/processed/task2/clinical_texts_task2.csv` ($70 \times 2$, columna `full_encounter_narrative`)
  - `data/processed/task2/ground_truth_task2.csv` ($70 \times 14$, columna `treatment_decision` y `confidence`)
- **Particiones**: Matriz estandarizada `data/processed/task2/mccv_splits_30.csv` ($70 \times 31$), con 30 particiones 80/20 (56 entrenamiento / 14 validación) estratificadas multiobjetivo.
- **Preprocesamiento por split (fold-isolated, sin data leakage)**:
  - **Tabular**: Imputación de medianas (numéricas) y modas (categóricas) de train + One-Hot Encoding + `StandardScaler` ajustado únicamente en train. Exclusión total de `active_treatment_flag`.
  - **MRI (1024D)**: `StandardScaler` ajustado únicamente sobre embeddings de entrenamiento.
  - **Texto**: `TfidfVectorizer` ajustado únicamente sobre las narrativas clínicas de entrenamiento.

## 3. File Layout for This Experiment

```
experiments/exp_16/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   └── run_exp16_decision_mccv30.py
├── results/
│   ├── aggregate_metrics.json
│   ├── split_metrics.json
│   ├── best_hyperparams.json
│   └── version.txt
└── reports/
    └── summary.md
```

## 4. Baselines & Proposed Conditions

| Condición | Modelo Base por Modalidad | Targets de Entrenamiento | Pesos Fusión Simplex $\Delta^2$ |
|---|---|---|---|
| **C1 (Hard-SVC Baseline)** | `SVC(kernel='linear', C=0.1/10.0, probability=True)` | Hard One-Hot Labels ($y_k \in \{0, 1\}$) | $w = (0.85, 0.15, 0.00)$ (Tab/MRI/Txt) |
| **C2 (Fuzzy-SVR Champion)** | `MultiOutputRegressor(LinearSVR(C=0.1/10.0, eps=0.05/0.1))` | Soft Targets Difusos ($y_k = 0.25 + 0.75c$) | $w = (0.55, 0.20, 0.25)$ (Tab/MRI/Txt) |

## 5. Evaluation Protocol (Gobernanza Estricta con `utils/pkg/chimera_metrics`)

Queda estrictamente prohibido implementar funciones ad-hoc de evaluación en el script del experimento. Todas las evaluaciones deben ejecutarse importando directamente las funciones del paquete `chimera_metrics`:

1. **Normalización y Compuerta de Decisión Oficial**:
   - `chimera_metrics.core.base.norm_treatment_decision`
   - `chimera_metrics.task2.decision.compute_treatment_decision_gate`
2. **Métricas Primarias**:
   - Macro-F1 promedio $\pm$ desviación estándar a través de los 30 splits MCCV.
   - Brier Score multiclase (calibración de probabilidades / salidas difusas normalizadas $L_1$).
3. **Métricas Secundarias**:
   - Micro-F1, Weighted-F1, AUROC OVR multiclase.
   - F1 por clase (`watchful_waiting`, `continued_surveillance`, `active_surveillance`, `active_treatment`).
   - Tasa de acierto de compuerta ($S_{gate}=1.0$).

## 6. Expected Results & Decision Rules

- **Criterio de Éxito**: Si C2 (Fuzzy-SVR) obtiene $\overline{\text{Macro-F1}} \ge 0.60$ y $\text{Brier} \le 0.40$ bajo MCCV-30, se confirma la replicabilidad del modelo champion de la Subtarea 2.1 para la integración en el agente CHIMERA.
- **Análisis de Discrepancias**: Se auditará el comportamiento en la clase minoritaria `watchful_waiting` ($n=1$ en cada fold de validación) y se contrastará con los resultados de MCCV-100 de `exp_11`.

## 7. Risks & Mitigations

- **Riesgo de Data Leakage**: Mitigado garantizando que escaladores (`StandardScaler`) y vectorizadores (`TfidfVectorizer`) se ajusten exclusivamente en las 56 muestras de entrenamiento de cada partición.
- **Riesgo de Divergencia de Evaluación**: Mitigado mediante el consumo obligatorio de `chimera_metrics.task2.decision`.

## 8. Reproducibility Checklist

- [ ] Semillas fijas y controladas por split en `data/processed/task2/mccv_splits_30.csv`.
- [ ] Versión del repositorio registrada en `results/version.txt`.
- [ ] Evaluación gobernada 100% por `utils/pkg/chimera_metrics`.
- [ ] Métricas por split y agregadas exportadas en formato JSON estructurado.

## 9. Next Steps

1. Revisar y aprobar este diseño experimental (`DESIGN.md`).
2. Generar y aprobar el plan de implementación (`IMPLEMENTATION.md`).
3. Ejecutar `scripts/run_exp16_decision_mccv30.py` y documentar los resultados en `reports/summary.md` mediante `experiment-reporter`.
