# Experiment Summary: exp_1 — Transformación Tabular Compartimentada y EDA Task 1

**Experiment**: `experiments/exp_1/`  
**Goal**: 02 — CHIMERA-Agent Phase 2  
**Date**: 2026-11-05  
**Status**: Complete  
**Verdict**: Success (Hypothesis Confirmed)  

---

## 1. Executive Summary

El Experimento `exp_1` implementó con éxito la extracción, normalización y compartimentación de los 195 casos de Task 1 (CHIMERA Fase 2) desde JSON no estructurados hacia dos tablas analíticas ([`../../data/inputs.csv`](file:///home/jmalagont/Documents/prostate-cancer-reasoning/goals/02/data/inputs.csv) con 195 filas y [`../../data/outputs.csv`](file:///home/jmalagont/Documents/prostate-cancer-reasoning/goals/02/data/outputs.csv) con 91 filas de Ground Truth).

El análisis exploratorio de datos (EDA) reveló los determinantes clínicos fundamentales de la decisión de biopsia, resolviendo de forma concluyente la aparente paradoja de pacientes con alto riesgo radiológico ($PIRADS \ge 4$) que no son biopsiados.

---

## 2. Hypothesis & Metric Validation

| Métrica / Criterio | Umbral de Diseño (`DESIGN.md`) | Resultado Obtenido | Veredicto |
|---|---|---|---|
| **Tasa de Recuperación de Variables Clave** | $\ge 98.0\%$ | **100.0%** (195/195 casos con PIRADS, volumen, PSAD, PSA, edad, DRE, bx, csPCa) | **Superado** |
| **Integridad de Ground Truth (`outputs.csv`)** | 0 nulos en las 15 columnas (91 casos) | **0 nulos** (100% completo en decisión, confianza, 10 pesos, secuencia y rationale) | **Superado** |
| **Caracterización de Discordancias Clínicas** | Identificación del 100% de casos $PIRADS \ge 4$ con decisión `"no"` | **100% identificados (24 casos)** y explicados fisiopatológicamente | **Superado** |

---

## 3. Hallazgos Clínicos y Descubrimientos Estadísticos Clave

### A. Matriz de Contingencia PI-RADS vs Decisión de Biopsia ($N=91$)

| PI-RADS | Decisión: `"no"` | Decisión: `"yes"` | Total | Tasa de Biopsia |
|---|---|---|---|---|
| **2** | 3 | 0 | 3 | **0.0%** |
| **3** | 8 | 2 | 10 | **20.0%** |
| **4** | 9 | 24 | 33 | **72.7%** |
| **5** | 15 | 30 | 45 | **66.7%** |
| **Total** | **35** | **56** | **91** | **61.5%** |

- **Regla Estricta para Bajo Riesgo:** Si $PIRADS \le 2$, la decisión es determinísticamente `"no"` ($0\%$ biopsias). Cero falsos positivos.
- **Zona Gris ($PIRADS = 3$):** Sólo el $20\%$ es biopsiado; la decisión depende de la densidad de PSA ($PSAD > 0.15$) o la velocidad de duplicación.

### B. El Mecanismo de Discordancia en Alto Riesgo ($PIRADS \ge 4$ con `"no"`)
De los 24 pacientes con $PIRADS \ge 4$ que no recibieron indicación de biopsia:
1. **$22$ de los $24$ casos ($91.7\%$) presentan `bx == "Positive"`:** Cuentan con un diagnóstico previo de cáncer confirmado en el historial clínico (ej. ISUP 3 o ISUP 4). Por guías urológicas, **no requieren una biopsia diagnóstica repetida**, sino que son derivados directamente a tratamiento curativo activo (prostatectomía/radioterapia) o terapia hormonal sistémica.
2. **Los 2 casos restantes:** Corresponden a pacientes añosos ($\ge 79$ años) con $PSAD$ muy bajo ($<0.03$), donde el riesgo/beneficio clínico desaconseja el procedimiento invasivo.

### C. Jerarquía Real de Pesos de Variables según Especialistas
Al analizar las 10 variables clínicas evaluadas en `variable_weights`:
- **`pirads`:** Decisivo en 45 casos, Importante en 45 casos (98.9% de relevancia alta).
- **`bx` (Historial de biopsia):** Decisivo en 18 casos, Importante en 46 casos.
- **`psa`:** Importante en 51 casos, Decisivo en 10 casos.
- **`psad`:** Importante en 25 casos, Decisivo en 5 casos, Anotado en 52 casos.
- **`cspca` (Deep Learning PI-CAI):** Clasificado como `not_used` en **61 de 91 casos (67.0%)**. Los clínicos se apoyan en imagenología y antecedentes histológicos, relegando el modelo de caja negra.

### D. Secuencia de Consulta de Herramientas (`reveal_sequence`)
Frecuencia de consulta en los 91 casos:
1. `radiology_report`: **88 / 91 (96.7%)**
2. `psa_trend`: **79 / 91 (86.8%)**
3. `previous_notes`: **77 / 91 (84.6%)**
4. `laboratory_results`: **41 / 91 (45.1%)**

---

## 4. Consolidated Artifacts

- **Datasets Limpios:**
  - [`goals/02/data/inputs.csv`](file:///home/jmalagont/Documents/prostate-cancer-reasoning/goals/02/data/inputs.csv) (195 filas, 22 columnas prefijadas por sección).
  - [`goals/02/data/outputs.csv`](file:///home/jmalagont/Documents/prostate-cancer-reasoning/goals/02/data/outputs.csv) (91 filas, 15 columnas con Ground Truth exhaustivo).
- **Métricas y Auditoría:**
  - [`results/dataset_audit.json`](file:///home/jmalagont/Documents/prostate-cancer-reasoning/goals/02/experiments/exp_1/results/dataset_audit.json)
  - [`results/statistical_summary.json`](file:///home/jmalagont/Documents/prostate-cancer-reasoning/goals/02/experiments/exp_1/results/statistical_summary.json)
  - [`results/version.txt`](file:///home/jmalagont/Documents/prostate-cancer-reasoning/goals/02/experiments/exp_1/results/version.txt)
- **Figuras Diagnósticas:**
  - [`reports/figures/pirads_vs_decision.png`](file:///home/jmalagont/Documents/prostate-cancer-reasoning/goals/02/experiments/exp_1/reports/figures/pirads_vs_decision.png)
  - [`reports/figures/distributions_by_class.png`](file:///home/jmalagont/Documents/prostate-cancer-reasoning/goals/02/experiments/exp_1/reports/figures/distributions_by_class.png)
  - [`reports/figures/tool_usage_and_weights.png`](file:///home/jmalagont/Documents/prostate-cancer-reasoning/goals/02/experiments/exp_1/reports/figures/tool_usage_and_weights.png)

---

## 5. Actionable Takeaways para `exp_2` (Inducción del Árbol Agéntico)

1. **Estructura Lógica del Árbol Neurosimbólico:**
   - La raíz debe consultar de inmediato `radiology_report`.
   - Si $PIRADS \le 2 \to$ Decisión `"no"`.
   - Si $PIRADS \ge 4 \to$ Evaluar antecedentes: si `structured_prompt__bx == "Positive"` y tiene diagnóstico previo documentado $\to$ Decisión `"no"` (tratamiento directo); si es naive $\to$ Decisión `"yes"`.
   - Si $PIRADS = 3 \to$ Rama de desempate consultando `radiology_report__psa_density` ($>0.15$) o `psa_trend`.
2. **Generación Determinista de Razonamiento:**
   - Los pesos de variables y la `reveal_sequence` se pueden deducir directamente de los nodos atravesados en esta estructura lógica sin costo de cómputo adicional.
