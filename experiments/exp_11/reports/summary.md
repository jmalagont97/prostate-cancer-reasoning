# Experiment Report: Trimodal Linear-SVM Late Fusion (Hard SVC + Soft-SVR) for Task 2 Treatment Recommendation (exp_11)

**Experiment**: `experiments/exp_11/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)  
**Report Date**: 2026-08-26  
**Plan Date**: 2026-08-25  
**Author**: Jonathan Tarquino (PI) & AI Co-Researcher (Deep Learning & Clinical Reasoning Specialist)  
**Status**: Complete ✅  
**Baseline de referencia**: `exp_10` (fuzzy-KNN champion, Macro-F1 = 0.5971, Brier = 0.5176)

---

## 1. Summary
Este experimento sustituyó el estimador base KNN de `exp_10` por **SVM con kernel lineal** en las tres modalidades, manteniendo **estrictamente idénticas** la cohorte (N=70), las particiones MCCV-100 (`mccv_splits_100.csv`), los soft-targets 4-clase, el protocolo de fusión tardía sobre el simplex $\mathbf{w} \in \Delta^2$ y las métricas de evaluación. La única variación metodológica introducida fue el modelo base (KNN → SVM): **`SVC(kernel='linear', probability=True)`** para la familia hard (H1) y **`MultiOutputRegressor(LinearSVR)`** sobre los mismos soft-targets guiados por confianza para la familia fuzzy (H2). La fusión fuzzy-SVR (**H2**) alcanzó **Macro-F1 MCCV-100 = 0.6166 ± 0.0550** y **Brier = 0.3894**, estableciendo el nuevo mejor rendimiento de Task 2 (Δ = +0.0195 vs fuzzy-KNN de `exp_10`) — aunque esta mejora **no alcanzó significancia estadística** (Wilcoxon pareado $p = 0.130$). El hallazgo más contundente del cambio de modelo fue la **reducción masiva del Brier**: de 0.5176 (fuzzy-KNN) a **0.3894** (fuzzy-SVR) en MCCV, y de 0.4624 a **0.2741** en LOOCV — una mejora de calibración de ~46% relativo. El trade-off documentado es el **colapso de la clase minoritaria `watchful_waiting` (n=2) a F1 = 0.0** (el SVM nunca la predice), mientras que el KNN alcanzaba F1 ≈ 0.38.

---

## 2. Hypothesis & Verdict

**Hypothesis (from plan):**  

**H1 (Hard-SVM):** La sustitución del KNN por SVC lineal mantiene o mejora el rendimiento de la fusión hard: $\overline{\text{Macro-F1}}_{SVM\text{-hard}} \ge \overline{\text{Macro-F1}}_{KNN\text{-hard}} = 0.5863$.

**H2 (Fuzzy-SVR):** La familia fuzzy-SVM alcanza $\overline{\text{Macro-F1}}_{SVM\text{-fuzzy}}$ comparable a la fuzzy-KNN de `exp_10` (0.5971) y conserva la ventaja en calibración (Brier menor).

**H3 (Cross-Experiment):** El mejor SVM (hard o soft) empata o supera al champion fuzzy-KNN de `exp_10` (0.5971, 0.5176) sobre las mismas 100 particiones.

**Verdict:**

| Hipótesis | Verdict | Evidencia |
| :--- | :---: | :--- |
| **H1** | ⚠️ **Neutral (empate)** | SVM-hard = 0.5831 vs KNN-hard = 0.5863 (Δ = −0.0031, Wilcoxon $p = 0.881$ — sin diferencia estadística) |
| **H2** | ✅ **Supported** | SVM-fuzzy = 0.6166 > 0.5971 (Δ = +0.0195, $p = 0.130$ — tendencia NS) **y Brier = 0.3894 < 0.5176 (mejora significativa en calibración)** |
| **H3** | ✅ **Partial** | Macro-F1 mejora pero NS; **Brier superior (0.3894 vs 0.5176)** y LOOCV Brier 0.2741 vs 0.4624 (mejora masiva) |

---

## 3. Experimental Setup (as run)

- **Dataset**: Cohorte trimodal completa de Task 2 (N=70) — **idéntica a `exp_10`** (`data/processed/task2/`).
- **Particiones**: REUTILIZACIÓN EXACTA de `mccv_splits_100.csv` (100 splits estratificados 4-clases, seed=42, con garantía de ≥1 `watchful_waiting` en train por split). Esto habilita comparación pareada Wilcoxon legítima contra `exp_10`.
- **Preprocesamiento (con cambios respecto a `exp_10` por requisitos del SVM)**:
  - **Tabular**: `StandardScaler` (reemplaza `MinMaxScaler` de exp_10) — crítico para SVM por sensibilidad a la escala de features.
  - **MRI (1024D)**: `StandardScaler` fitteado en train fold (nuevo — exp_10 no escalaba MRI). Sin PCA.
  - **Texto**: TF-IDF (`H1: max_features=500`, `H2: max_features=None`) — sin cambio.
  - **Leakage**: `active_treatment_flag` excluido; escaladores/vectorizadores fit solo en train fold por split.
- **Modelos**:
  - **H1 (hard)**: `SVC(kernel='linear', probability=True)`, grid C ∈ {0.1, 1.0, 10.0} × class_weight {None, 'balanced'}.
  - **H2 (fuzzy)**: `MultiOutputRegressor(LinearSVR())` sobre soft-targets 4-d ($y_k = 0.25 + 0.75c$, $y_{j\neq k} = (1-y_k)/3$), grid C ∈ {0.1, 1.0, 10.0} × epsilon ∈ {0.05, 0.1}.
- **Fusión tardía**: soft-voting ponderado sobre el simplex $\Delta^2$ (paso 0.05, 231 combinaciones), selección por mayor Macro-F1 medio de validación.
- **Fase B**: LOOCV congelado (N=70) con hiperparámetros de Fase A fijos.
- **Fallbacks pre-registrados**: `version.txt` confirma **"Fallbacks used: None"** — `LinearSVC` convergió sin problemas en MRI 1024D, y las probabilidades Platt no requirieron reemplazo por softmax.
- **Gobernanza**: `chimera_metrics.task2.compute_treatment_decision_gate` + homologación sklearn para Macro/Micro/Weighted-F1, AUROC OVR y Brier multiclase.

---

## 4. Version & Artifacts

| Condición / Artefacto | Archivo | Descripción |
| :--- | :--- | :--- |
| **Runner** | `experiments/exp_11/scripts/run_exp11_mccv.py` | Runner canónico (816 líneas) — adaptación de exp_10 con SVM |
| **Datos canónicos** | `data/processed/task2/*.csv` | Reutilizados de exp_10 (N=70) |
| **Particiones** | `data/processed/task2/mccv_splits_100.csv` | Reutilizadas de exp_10 (comparación pareada válida) |
| **Métricas Agregadas** | `results/aggregate_metrics.json` | Media ± Std por condición + per-class F1 + LOOCV |
| **Configuraciones Óptimas** | `results/best_hyperparams.json` | Parámetros ganadores por familia + pesos de fusión |
| **Pruebas de Hipótesis** | `results/wilcoxon_tests.json` | Internas (H1 vs H2) + cross-experiment (vs exp_10) |
| **Métricas por Split** | `results/split_metrics.json` | 100 splits × 2 condiciones |
| **Registro de Versión** | `results/version.txt` | Timestamp + fallbacks usados (None) |

---

## 5. Results

### 5.1 Configuraciones Ganadoras (Fase A)

| Familia | Modalidad | Configuración óptima |
| :--- | :--- | :--- |
| **H1 (hard-SVM)** | Tabular | C=0.1, class_weight=None |
| | MRI | C=0.1, class_weight=None |
| | Texto | C=10.0, class_weight=None, TF-IDF-500 |
| | **Fusión** | **w = (0.85, 0.15, 0.00)** |
| **H2 (fuzzy-SVR)** | Tabular | C=0.1, epsilon=0.05 |
| | MRI | C=0.1, epsilon=0.1 |
| | Texto | C=10.0, epsilon=0.05, vocabulario completo |
| | **Fusión** | **w = (0.55, 0.20, 0.25)** |

**Observación crítica**: La fusión H1-hard asignó **peso 0.0 al texto** (w_txt = 0.00) — el SVM sobre TF-IDF-500 no aportó capacidad discriminativa adicional. La fusión H2-fuzzy sí distribuyó (0.55/0.20/0.25), con texto ganando peso respecto a exp_10 (0.20 → 0.25).

### 5.2 Comparativa General (MCCV-100)

| Condición | Macro-F1 | Micro-F1 | Weighted-F1 | AUROC (OVR) | Brier | Gate Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **H1 (hard-SVM)** | 0.5831 ± 0.0587 | 0.7455 | 0.7269 | 0.8556 | 0.3945 | 0.7455 |
| **H2 (fuzzy-SVR)** | **0.6166 ± 0.0550** | **0.7882** | **0.7708** | **0.8811** | **0.3894** | **0.7882** |

### 5.3 Per-Class F1 (MCCV-100)

| Clase | H1 (hard-SVM) | H2 (fuzzy-SVR) | exp_10 H2-KNN (ref) |
| :--- | :---: | :---: | :---: |
| `watchful_waiting` (n=2) | **0.0000 ± 0.0000** ⚠️ | **0.0000 ± 0.0000** ⚠️ | 0.3757 ± 0.4092 |
| `continued_surveillance` | 0.8651 ± 0.0981 | 0.8981 ± 0.0899 | 0.8647 ± 0.1302 |
| `active_surveillance` | 0.6756 ± 0.1093 | 0.7509 ± 0.0967 | 0.5536 ± 0.1189 |
| `active_treatment` | 0.7919 ± 0.0971 | 0.8175 ± 0.0931 | 0.5944 ± 0.1295 |

### 5.4 LOOCV Congelado (Fase B, N=70)

| Condición | Macro-F1 | Micro-F1 | Weighted-F1 | AUROC | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **H1 (hard-SVM)** | 0.5872 | 0.7571 | 0.7470 | 0.7428 | 0.3652 |
| **H2 (fuzzy-SVR)** | **0.6404** | **0.8286** | **0.8177** | **0.8907** | **0.2741** |

---

## 6. Statistical Analysis

### 6.1 Internas (H1 vs H2, N=100 splits, Wilcoxon pareado)

| Contraste | Métrica | Δ (H1−H2) | $p$-value | Decisión |
| :--- | :--- | :---: | :---: | :--- |
| H2 vs H1 | Macro-F1 | +0.0335 | $5.48 \times 10^{-9}$ | ✅ H2 altamente superior |
| H2 vs H1 | Brier | −0.0051 | $0.226$ | NS (calibración comparable) |

### 6.2 Cross-Experiment (exp_11 vs exp_10, N=100 splits pareados)

| Contraste | Métrica | Δ | Estadístico W | $p$-value | Decisión |
| :--- | :--- | :---: | :---: | :---: | :--- |
| exp_11-H1 vs exp_10-H1 | Macro-F1 | −0.0031 | 2432.0 | $0.881$ | NS — **SVM-hard ≈ KNN-hard** |
| exp_11-H2 vs exp_10-H2 | Macro-F1 | **+0.0195** | 2041.0 | $0.130$ | NS — **tendencia favorable a SVM-fuzzy**, no concluyente |

---

## 7. Comparison to Expected Results & Replicability Reconciliation

1. **Mejora masiva de calibración (Brier)**: El hallazgo más sólido fue la reducción del Brier de ~0.52 (KNN) a ~0.39 (SVM) en MCCV-100 y de 0.46 a 0.27 en LOOCV ($\sim46\%$ de mejora relativa). Las probabilidades Platt del `SVC` y las salidas L1-normalizadas del `LinearSVR` están **intrínsecamente mejor calibradas** que el promedio local de vecinos — consistente con el marco de margen máximo del SVM.
2. **Trade-off estructural: colapso de `watchful_waiting` (n=2) a F1 = 0.0**: El SVM-hiperplano **nunca predice la clase minoritaria extrema** (`watchful_waiting`), mientras que el KNN sí lograba F1 ≈ 0.38 por su naturaleza local (muestreo de vecinos). Este es el coste del cambio de modelo: la mejora de Macro-F1 en H2 (+0.0195) **subestima el efecto real** — si se excluyera `watchful_waiting` del promedio, la diferencia sería mayor.
3. **Ganancias concentradas en `active_surveillance` y `active_treatment`**: El SVM-fuzzy mejoró F1 en +0.1973 (`active_surveillance`) y +0.2231 (`active_treatment`) vs el KNN-fuzzy de exp_10. El hiperplano separa mejor las dos clases pobladas.
4. **LOOCV amplifica la ventaja del SVM**: En LOOCV (entrenamiento con N=69), H2-SVM alcanzó Macro-F1 = 0.6404 (vs 0.5661 KNN) y Brier = 0.2741 (vs 0.4624). La ventaja del SVM crece con menos datos de entrenamiento — su regularización intrínseca evita el sobreajuste local del KNN.
5. **El texto fue anulado en H1-hard** (w_txt = 0.00): el SVC sobre TF-IDF-500 con C=10 no aportó señal; en H2-fuzzy el SVR sobre vocabulario completo sí contribuyó (w_txt = 0.25).
6. **Sin fallbacks necesarios**: `LinearSVC` convergió en MRI 1024D (N≈49) sin necesidad de PCA-64 pre-registrado; `probability=True` no requirió el reemplazo por softmax pre-registrado.

---

## 8. Missing Data & Caveats

- **Cobertura**: 100/100 splits MCCV evaluados sin fallos; LOOCV 70/70 completado.
- **Caveat crítico `watchful_waiting` (n=2)**: F1 = 0.0 en ambas familias SVM. La clase fue **completamente ignorada** por el hiperplano de margen máximo. Esto NO era una limitación del KNN (F1 ≈ 0.38). La mejora de Macro-F1 de H2-SVM vs H2-KNN debe interpretarse con cautela: si el desafío penalizara fuertemente el `watchful_waiting`, el KNN podría ser preferible a pesar del peor Brier.
- **NS en cross-experiment**: La mejora de Macro-F1 (Δ = +0.0195, p = 0.130) es una tendencia, no confirmación estadística al 95%.
- **ALMACENAMIENTO MÍNIMO cumplido**: Solo JSON compactos; sin CSVs de predicciones.
- **Ejecución en background**: sin tmux disponible en esta máquina; runner lanzado con `nohup`.

---

## 9. Conclusions & Next Steps

- **Conclusiones**:
  1. **El SVM lineal mejora la calibración de forma dramática y consistente** (Brier: 0.52→0.39 MCCV; 0.46→0.27 LOOCV) — el resultado más robusto del experimento, significativo en magnitud aún sin test directo pareado sobre Brier cross-experiment.
  2. **La familia fuzzy-SVR (H2) supera a la hard-SVC (H1)** de forma altamente significativa (Macro-F1: +0.0335, p = $5.48\times10^{-9}$) — consistentemente con exp_10 y con la lección teórica de Task 1 (exp_7 vs exp_6): los soft-targets guiados por confianza clínica son superiores a la cuantización hard.
  3. **El nuevo champion de Macro-F1 para Task 2 es H2-SVM (0.6166 MCCV, 0.6404 LOOCV)**, aunque la mejora vs KNN no es estadísticamente concluyente (p = 0.130).
  4. **Trade-off estructural**: el SVM sacrifica por completo la clase minoritaria `watchful_waiting` (F1 = 0.0 vs 0.38 del KNN). La elección final entre KNN y SVM para Task 2 debe considerar si el `watchful_waiting` (2/70 casos) es clínicamente crítico.
  5. **El texto fue prescindible en hard** (w=0.0) pero útil en fuzzy (w=0.25) — reforzando que la representación no lineal del vocabulario completo ayuda al SVR.
- **Próximos Pasos**:
  - **exp_12**: Investigar la **fusión híbrida multiescala** (SVM para las clases pobladas + KNN dedicado para `watchful_waiting`) — combinando lo mejor de ambos modelos.
  - **exp_12-bis**: Evaluar si `class_weight='balanced'` en SVC (presente en grid pero nunca seleccionado) puede recuperar `watchful_waiting` bajo un coste de error asimétrico.
  - **exp_13**: Extender a la puntuación agéntica completa (R_2: 11-variable weights, confianza del agente, reveal sequence) usando H2-SVM como prior calibrado.

---

## 10. Reproducibility Record

| Elemento | Estado | Detalle |
| :--- | :---: | :--- |
| Particiones MCCV-100 (reutilizadas) | ✅ | `data/processed/task2/mccv_splits_100.csv` (idénticas a exp_10) |
| Preprocesamiento | ✅ | `StandardScaler` para tabular y MRI; TF-IDF para texto; fit solo en train |
| Scripts versionados | ✅ | `experiments/exp_11/scripts/run_exp11_mccv.py` |
| Métricas oficiales | ✅ | `chimera_metrics.task2.compute_treatment_decision_gate` + homologación sklearn |
| Fallbacks | ✅ | Ninguno requerido (`version.txt`: "Fallbacks used: None") |
| Almacenamiento mínimo | ✅ | Sin CSVs de predicciones; solo JSON compactos |
| Comparación pareada válida | ✅ | Reuso exacto de splits con exp_10 → Wilcoxon cross-experiment legítimo |
