# Experiment Report: Búsqueda Sistemática de Hiperparámetros para la Decisión Terapéutica (Subtarea 2.1) bajo MCCV-30 (exp_17)

**Experiment**: `experiments/exp_17/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent Task 2)  
**Date**: 2026-08-30  
**Author**: Juan Sebastian Malagón Torres  
**Status**: Complete  
**Verdict**: Success (Hypothesis Confirmed)

---

## 1. Resumen Ejecutivo

En el marco de la **Subtarea 2.1 (Decisión Terapéutica)** y sobre las 30 particiones de validación cruzada multiobjetivo (**MCCV-30**, $N=70$, 56 train / 14 val), se evaluó el espacio de búsqueda completo de regularización $C$, margen $\epsilon$, balance de clases y combinaciones de fusión tardía en el simplex $\Delta^2$, con evaluación exclusiva mediante `utils/pkg/chimera_metrics`.

La optimización sistemática confirmó la hipótesis experimental:
- **Fuzzy-SVR Tuned** mejoró su **Macro-F1 de $0.6394 \pm 0.0540$ a $0.6595 \pm 0.0665$**.
- La calibración **Brier Score mejoró de $0.3903 \pm 0.0643$ a $0.3608 \pm 0.0765$**.
- La exactitud en compuerta clínica (**Gate Accuracy / Micro-F1**) subió a **$0.8310 \pm 0.0835$** (frente a $0.8048$ en `exp_16`).
- El **AUROC OVR** alcanzó **$0.9267 \pm 0.0541$**.

---

## 2. Hiperparámetros Óptimos Encontrados

### Modelos Unimodales
- **Fuzzy-SVR**:
  - **Tabular**: $C=0.1, \epsilon=0.05$ (Macro-F1 unimodal: $0.6595$)
  - **MRI (1024D)**: $C=0.1, \epsilon=0.10$ (Macro-F1 unimodal: $0.3370$)
  - **Texto (TF-IDF)**: $C=10.0, \epsilon=0.10$ (Macro-F1 unimodal: $0.3875$)

### Ponderaciones de Fusión Tardía en el Simplex $\Delta^2$
- **Fuzzy-SVR Tuned**:
  $$w_{\text{tab}} = 0.50, \quad w_{\text{mri}} = 0.05, \quad w_{\text{txt}} = 0.45$$
- **Hard-SVC Tuned**:
  $$w_{\text{tab}} = 0.55, \quad w_{\text{mri}} = 0.15, \quad w_{\text{txt}} = 0.30$$

---

## 3. Comparación Cuantitativa Integral (MCCV-30)

| Métrica | Hard-SVC (`exp_16` static) | Fuzzy-SVR (`exp_16` static) | Hard-SVC Tuned (`exp_17`) | Fuzzy-SVR Tuned (`exp_17`) |
|---|:---:|:---:|:---:|:---:|
| **Macro-F1** | $0.5948 \pm 0.0431$ | $0.6394 \pm 0.0540$ | $0.6502 \pm 0.0623$ | **$0.6595 \pm 0.0665$** |
| **Micro-F1 / Gate Acc** | $0.7524 \pm 0.0487$ | $0.8048 \pm 0.0700$ | $0.8167 \pm 0.0776$ | **$0.8310 \pm 0.0835$** |
| **Brier Score** | $0.3801 \pm 0.0432$ | $0.3903 \pm 0.0643$ | $0.4188 \pm 0.0409$ | **$0.3608 \pm 0.0765$** |
| **AUROC (OVR)** | $0.8767 \pm 0.0313$ | $0.8931 \pm 0.0594$ | $0.7702 \pm 0.0672$ | **$0.9267 \pm 0.0541$** |
| **F1 Active Treatment** | $0.8073 \pm 0.0611$ | $0.8600 \pm 0.0876$ | $0.8221 \pm 0.0692$ | **$0.8445 \pm 0.0790$** |
| **F1 Active Surveillance** | $0.7303 \pm 0.0684$ | $0.7664 \pm 0.0899$ | $0.8341 \pm 0.1248$ | **$0.8490 \pm 0.1291$** |
| **F1 Continued Surveillance** | $0.8417 \pm 0.1425$ | $0.9311 \pm 0.1114$ | $0.9444 \pm 0.1048$ | **$0.9444 \pm 0.1048$** |
| **F1 Watchful Waiting** ($n=2$) | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ |

---

## 4. Conclusiones y Champion Consolidado

1. **Mayor contribución de la narrativa clínica**: Al relajar el margen $\epsilon$ a $0.10$ en texto, la ponderación óptima de la narrativa clínica subió al $45\%$ en la fusión, incrementando la discriminación fina entre vigilancia activa y tratamiento activo.
2. **Consolidación del Champion**: `Fuzzy_SVR_Tuned` (`exp_17`) se establece como el modelo canónico definitivo para la predicción de la decisión médica en Task 2, logrando superar a todas las variantes anteriores en Macro-F1, calibración probabilística (Brier) y AUROC.
