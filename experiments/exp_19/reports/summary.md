# Experiment Report: Búsqueda Sistemática de Hiperparámetros y Meta-Modelos por Variable para Relevancia Clínica (Subtarea 2.3) bajo MCCV-30 (exp_19)

**Experiment**: `experiments/exp_19/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent Challenge MICCAI 2026)  
**Date**: 2026-08-30  
**Author**: Sebastian Malagon  
**Status**: Complete  
**Verdict**: Baseline C1 (`exp_12` ExtraTrees/RF unificado $n=200$, `balanced`) es estadísticamente y clínicamente superior frente a C2.

---

## 1. Resumen Ejecutivo
Se implementó y ejecutó el benchmark de la **Subtarea 2.3 (Ponderación de Relevancia Clínica para 11 variables urológicas)** sobre el protocolo de validación cruzada multiobjetivo **MCCV-30** (`data/processed/task2/mccv_splits_30.csv`), evaluando rigurosamente con las funciones canónicas de `chimera_metrics.task1.reasoning` (`compute_variable_weights_score` y `compute_important_decisive_f1`).

---

## 2. Resultados Comparativos Oficiales (MCCV-30 con `chimera_metrics`)

| Métrica | C1: Baseline (`exp_12` $n=200$, `balanced`) | C2: Per-Variable Tuned (`exp_19`) | Diferencia ($\Delta$) | p-valor (Wilcoxon) |
|---|:---:|:---:|:---:|:---:|
| **$S_{\text{weights}}$ (Score Ordinal MAE)** | **$0.8307 \pm 0.0142$** | $0.8087 \pm 0.0170$ | $-0.0220$ | $p < 10^{-6}$ |
| **Set-$F_1$ (Factores Decisivos / Importantes)** | **$0.6826 \pm 0.0325$** | $0.6316 \pm 0.0438$ | $-0.0510$ | $p < 10^{-6}$ |

---

## 3. Análisis Metodológico y Causa del Hallazgo
1. **Regularización y Dispersión del Ensamble**:
   El baseline C1 (`exp_12`) utiliza $n_{\text{estimators}}=200$ con partición aleatoria extrema (`ExtraTreesClassifier`) y `class_weight='balanced'`. Esto provee una regularización estocástica superior al promediar 200 árboles sobre espacios de características de 98D (Tabular 32D + MRI PCA 16D + Texto TF-IDF 50D).
2. **Impacto de la Optimización Aislada en Macro-F1**:
   Al sintonizar meta-modelos optimizando localmente Macro-F1 por variable con $n=100$, varias variables seleccionaron `class_weight=None` (e.g. `age`, `bx_gl_sec`, `comorbidity`, `ct`), lo que provocó una subestimación de las clases minoritarias pero críticas (`important` y `decisive`). Esto degradó directamente el Set-$F_1$ oficial de factores clínicos ($0.6826 \to 0.6316$).

---

## 4. Conclusión y Consolidación del Modelo Champion
El **Baseline C1 (`exp_12`)** queda ratificado formalmente como el modelo **Champion** para la Subtarea 2.3:
- Configuración: Meta-clasificadores `ExtraTrees` (para 9 variables) y `RandomForest` (para `bx_isup` y `ct`) con $n_{\text{estimators}}=200$, `class_weight='balanced'`, y `max_features='sqrt'`.
- Garantiza la máxima puntuación en $S_{\text{weights}}$ ($0.8307$) y máxima fidelidad de factores decisivos ($0.6826$).
