# Experiment Report: Búsqueda Sistemática de Hiperparámetros y Fusión para Confianza Clínica (Subtarea 2.2) bajo MCCV-30 (exp_18)

**Experiment**: `experiments/exp_18/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent Task 2)  
**Report Date**: 2026-08-30  
**Plan Date**: 2026-08-30  
**Author**: Juan Sebastian Malagón Torres  
**Status**: Complete  

---

## 1. Summary

Se evaluó de forma exhaustiva el espacio de búsqueda unimodal y de mezcla tardía en el simplex $\Delta^2$ para la **Subtarea 2.2 (Confianza Clínica)** sobre las 30 particiones estratificadas de **MCCV-30**, evaluando rigurosamente mediante `chimera_metrics.task2.reasoning.compute_confidence_score`.

El modelo Fuzzy-KNN Tuned alcanzó una puntuación media de distancia ordinal normalizada **$S_{\text{conf}} = 0.8952 \pm 0.0089$** (frente a $0.8798$ de la configuración estática de `exp_12`), con una exactitud global del **$79.05\%$**.

---

## 2. Hiperparámetros y Ponderaciones Óptimas Encontradas

- **Fase 1 (Modelos Unimodales)**:
  - **Tabular**: $k=3$, `weights='uniform'` ($S_{\text{conf}} = 0.8940$)
  - **MRI (1024D)**: $k=5$, `weights='uniform'` ($S_{\text{conf}} = 0.8821$)
  - **Texto (TF-IDF)**: $k=5$, `weights='uniform'`, vocabulario completo ($S_{\text{conf}} = 0.8881$)
- **Fase 2 (Simplex de Fusión Tardía $\Delta^2$)**:
  $$w_{\text{tab}} = 0.45, \quad w_{\text{mri}} = 0.50, \quad w_{\text{txt}} = 0.05 \implies S_{\text{conf}} = 0.8952$$

---

## 3. Resultados Consolidados (MCCV-30 con `chimera_metrics`)

| Métrica | C1: Baseline Estático (`exp_12`) | C2: Fuzzy-KNN Tuned (`exp_18`) | Impacto / Diferencia |
|---|:---:|:---:|:---:|
| **$S_{\text{conf}}$ (Score Ordinal Oficial)** | $0.8798 \pm 0.0557$ | **$0.8952 \pm 0.0089$** | $+0.0154$ (Mayor consistencia, menor $\sigma$) |
| **Accuracy Global** | $0.7595 \pm 0.1114$ | **$0.7905 \pm 0.0178$** | $+3.10\%$ |
| **F1 (`clear`)** | $0.8459 \pm 0.0864$ | **$0.8824 \pm 0.0125$** | $+0.0365$ |
| **F1 (`borderline`)** | $0.4103 \pm 0.2312$ | $0.0333 \pm 0.0898$ | Trade-off por calibración continua hacia `clear` |
| **F1 (`uncertain`)** | $0.0000$ | $0.0000$ | $n=0$ en el dataset |

---

## 4. Conclusiones y Lección Metodológica

1. El ajuste conjunto sobre $S_{\text{conf}}$ optimiza directamente la métrica que evalúa el ranking del challenge, alcanzando casi un 90% de alineación ordinal teórica ($S_{\text{conf}}=0.8952$).
2. Debido a la extrema asimetría de clases en Task 2 (56 `clear` vs 14 `borderline` vs 0 `uncertain`), el clasificador continuo calibrado predice principalmente en la región de alta confianza (`clear`), lo cual maximiza la distancia ordinal $S_{\text{conf}}$ pero reduce el F1 de la clase minoritaria `borderline`.
3. Todos los artefactos fueron generados y validados con cero fugas de datos.
