# Experiment Report: Replicación de Fusión Trimodal Fuzzy-SVR para Decisión Terapéutica (Subtarea 2.1) bajo MCCV-30 (exp_16)

**Experiment**: `experiments/exp_16/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)  
**Report Date**: 2026-08-30  
**Plan Date**: 2026-08-30  
**Author**: Juan Sebastian Malagón Torres  
**Status**: Complete ✅  

---

## 1. Summary

Se replicó con éxito la arquitectura champion de la Subtarea 2.1 (fusión tardía trimodal **Fuzzy-SVR lineal** de `exp_11`) evaluada bajo el protocolo canónico multiobjetivo **MCCV-30** (`data/processed/task2/mccv_splits_30.csv`) y gobernada exclusivamente por el paquete oficial `chimera_metrics`.

Bajo MCCV-30, la fusión difusa **Fuzzy-SVR (C2)** confirmó su superioridad frente al baseline **Hard-SVC (C1)**:
- **Macro-F1**: **$0.6394 \pm 0.0540$** (C2) frente a **$0.5948 \pm 0.0431$** (C1), una mejora de $+0.0446$.
- **Gate Accuracy ($S_{gate} = 1.0$)**: **$0.8048 \pm 0.0700$** (C2) frente a **$0.7524 \pm 0.0487$** (C1).
- **Calibración (Brier Score)**: **$0.3903 \pm 0.0643$**, cumpliendo el criterio de éxito predefinido ($\le 0.40$).
- **AUROC OVR multiclase**: **$0.8931 \pm 0.0594$** (C2) vs **$0.8767 \pm 0.0313$** (C1).

---

## 2. Hypothesis & Verdict

| Hipótesis | Verdict | Evidencia Cuantitativa |
|---|:---:|---|
| **H1 (Replicación y Superioridad de Fuzzy-SVR)**: Fuzzy-SVR mantiene $\text{Macro-F1} \ge 0.60$ y $\text{Brier} \le 0.40$ bajo MCCV-30 multi-subtarea con gobernanza `chimera_metrics`. | ✅ **Supported** | $\overline{\text{Macro-F1}} = 0.6394$ ($\Delta = +0.0394$ vs umbral), $\text{Brier} = 0.3903 \le 0.40$. Supera a Hard-SVC en Macro-F1 ($0.6394$ vs $0.5948$) y Gate Accuracy ($80.48\%$ vs $75.24\%$). |

---

## 3. Detailed Results (MCCV-30)

### 3.1 Métricas Globales

| Condición | Macro-F1 | Micro-F1 / Gate Acc | Weighted-F1 | AUROC (OVR) | Brier Score |
|---|:---:|:---:|:---:|:---:|:---:|
| **C1 (Hard-SVC Baseline)** | $0.5948 \pm 0.0431$ | $0.7524 \pm 0.0487$ | $0.7214 \pm 0.0481$ | $0.8767 \pm 0.0313$ | $0.3801 \pm 0.0432$ |
| **C2 (Fuzzy-SVR Champion)** | **$0.6394 \pm 0.0540$** | **$0.8048 \pm 0.0700$** | **$0.7753 \pm 0.0684$** | **$0.8931 \pm 0.0594$** | **$0.3903 \pm 0.0643$** |

### 3.2 Rendimiento por Clase (F1-score en Validación)

| Clase Clínico-Terapéutica | C1 (Hard-SVC) | C2 (Fuzzy-SVR Champion) |
|---|:---:|:---:|
| `active_treatment` | $0.8368 \pm 0.0598$ | **$0.8600 \pm 0.0876$** |
| `active_surveillance` | $0.6646 \pm 0.0716$ | **$0.7664 \pm 0.0899$** |
| `continued_surveillance` | $0.8778 \pm 0.1486$ | **$0.9311 \pm 0.1114$** |
| `watchful_waiting` ($n=2$ cohorte total, $1$ en val) | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ |

---

## 4. Key Findings & Theoretical Insights

1. **Robustez de la Fusión Trimodal Difusa**:
   La regularización continua de `MultiOutputRegressor(LinearSVR)` sobre soft targets ponderados por la confianza médica aprovecha de forma óptima la combinación $55\%$ Tabular $+ 20\%$ MRI $+ 25\%$ Texto.
2. **Impacto en la Compuerta de Decisión ($S_{gate}$)**:
   Alcanzar más del $80.48\%$ de coincidencia exacta en decisión sobre particiones no vistas garantiza que el agente multimodal no sufra penalización de compuerta ($S_{gate}=0$) en la gran mayoría de casos de Task 2.
3. **Persistencia del Desafío de `watchful_waiting`**:
   Tal como se documentó en `exp_11`, los modelos lineales basados en hiperplanos colapsan la predicción de la clase de mínima frecuencia extrema ($n=2$), concentrando toda su masa en las 3 clases prevalentes.

---

## 5. Artifacts & Reproducibility Record

| Artefacto | Ruta |
|---|---|
| Particiones MCCV-30 | `data/processed/task2/mccv_splits_30.csv` |
| Runner Estandarizado | `experiments/exp_16/scripts/run_exp16_decision_mccv30.py` |
| Métricas Agregadas | `experiments/exp_16/results/aggregate_metrics.json` |
| Métricas por Split | `experiments/exp_16/results/split_metrics.json` |
| Version Tracking | `experiments/exp_16/results/version.txt` |
