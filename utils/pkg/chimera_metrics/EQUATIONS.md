# Formal Equations and Mathematical Specification: `chimera_metrics`

Este documento contiene la derivación y especificación matemática de todas las métricas implementadas en el paquete `chimera_metrics` para el reto **CHIMERA-Agent (MICCAI 2026)**, mapeando cada formulación a su implementación exacta en el código.

---

## 1. Tarea 1 y Tarea 2: Compuertas de Decisión Clínica (*Decision Gates*)

### 1.1 Decisión Binaria de Biopsia (Tarea 1)
Evalúa la concordancia exacta en la indicación pre-biopsia.
* **Ecuación:**
  ```
  S_gate(y_hat, y) = 1.0  si y_hat == y,  0.0  si y_hat != y
  ```
  donde $y, \hat{y} \in \{\text{yes}, \text{no}\}$.
* **Implementación:**
  * Función: `chimera_metrics.task1.compute_biopsy_decision_gate`
  * Clase: `chimera_metrics.task1.BiopsyDecisionGateMetric`
  * Referencia oficial: `evaluate.py::decision_score` (L706)

### 1.2 Recomendación Multiclase de Tratamiento (Tarea 2)
Evalúa la concordancia exacta en la categoría terapéutica post-biopsia.
* **Ecuación:**
  ```
  S_gate(y_hat, y) = 1.0  si y_hat == y,  0.0  si y_hat != y
  ```
  donde $y, \hat{y} \in \{\text{watchful\_waiting}, \text{active\_surveillance}, \text{continued\_surveillance}, \text{active\_treatment}\}$.
* **Implementación:**
  * Función: `chimera_metrics.task2.compute_treatment_decision_gate`
  * Clase: `chimera_metrics.task2.TreatmentDecisionGateMetric`
  * Referencia oficial: `evaluate.py::decision_score` (L697)

---

## 2. Razonamiento Clínico Determinista (Tareas 1 y 2)

### 2.1 Puntaje de Calibración de Confianza Ordinal
Mide la distancia ordinal absoluta normalizada sobre la escala cualitativa de confianza.
* **Ecuación:**
  ```
  S_conf(c_hat, c) = 1.0 - ( |rank(c_hat) - rank(c)| / K_conf )
  ```
  donde $\text{rank}(c) \in \{\text{uncertain}: 0, \text{borderline}: 1, \text{clear}: 2\}$ y $K_{\text{conf}} = 2$. Rango: $S_{\text{conf}} \in [0.0, 1.0]$.
* **Implementación:**
  * Función: `chimera_metrics.task1.compute_confidence_score` / `chimera_metrics.task2.compute_confidence_score`
  * Referencia oficial: `evaluate.py::confidence_score` (L719)

### 2.2 Ponderación Multivariable de Importancia Clínica (MAE Inverso)
Evalúa el error medio absoluto ordinal sobre el vector de variables clínicas consideradas.
* **Ecuación:**
  ```
  S_weights(W_hat, W) = 1.0 - ( 1 / |V| ) * sum_{v in V} [ |rank(W_hat(v)) - rank(W(v))| / K_weight ]
  ```
  donde $V$ es el conjunto de variables del ground truth, $\text{rank}(w) \in \{\text{not\_used}: 0, \text{noted}: 1, \text{important}: 2, \text{decisive}: 3\}$, $K_{\text{weight}} = 3$. Las variables omitidas en la predicción se imputan por defecto como $\text{not\_used}$ ($\text{rank}=0$). Rango: $S_{\text{weights}} \in [0.0, 1.0]$.
* **Implementación:**
  * Función: `chimera_metrics.task1.compute_variable_weights_score` / `chimera_metrics.task2.compute_variable_weights_score`
  * Referencia oficial: `evaluate.py::variable_weight_score` (L729)

### 2.3 Identificación de Factores Críticos (Set-F1 Score)
Evalúa la precisión y exhaustividad en la selección de variables determinantes (`important` o `decisive`).
* **Ecuación:**
  ```
  P_set = { v in V | rank(W_hat(v)) >= 2 }
  G_set = { v in V | rank(W(v)) >= 2 }

  F1(P_set, G_set) = 2 * |P_set inter G_set| / ( |P_set| + |G_set| )
  ```
  Condición de borde: si $P_{\text{set}} = \emptyset$ y $G_{\text{set}} = \emptyset \implies F_1 = 1.0$. Rango: $F_1 \in [0.0, 1.0]$.
* **Implementación:**
  * Función: `chimera_metrics.task1.compute_important_decisive_f1` / `chimera_metrics.task1.compute_set_f1`
  * Referencia oficial: `evaluate.py::important_decisive_factor_score` (L772)

---

## 3. Eficiencia y Anclaje Agéntico en el Historial Clínico

### 3.1 Eficiencia de Consulta de Herramientas (*Cost-Aware Tool Precision*)
Penaliza la inspección de información redundante no utilizada por los especialistas de referencia.
* **Ecuación:**
  ```
  S_tool(S_pred, S_gt) = |S_pred inter S_gt| / |S_pred|  si |S_pred| > 0,  1.0  si |S_pred| == 0
  ```
  donde $S_{\text{pred}}, S_{\text{gt}} \subseteq \{\text{previous\_notes}, \text{laboratory\_results}, \text{psa\_trend}, \text{radiology\_report}, \text{pathology\_report}, \text{family\_history}\}$. Rango: $S_{\text{tool}} \in [0.0, 1.0]$.
* **Implementación:**
  * Función: `chimera_metrics.task1.compute_cost_aware_tool_efficiency`
  * Clase: `chimera_metrics.task1.ToolEfficiencyMetric`
  * Referencia oficial: `evaluate.py::cost_aware_tool_score` (L1059)

### 3.2 Anclaje de Secciones (*Section Grounding*)
Penaliza al agente si asigna ponderación clínica activa ($\text{rank} \ge 1$) a variables cuya sección médica no fue revelada.
* **Ecuación:**
  ```
  S_grounding = N_grounded / ( N_grounded + N_ungrounded )
  ```
  donde $N_{\text{grounded}}$ cuenta variables pertenecientes a secciones consultadas (o disponibles en la tarjeta basal del paciente), y $N_{\text{ungrounded}}$ cuenta variables ponderadas sin evidencia consultada. Rango: $S_{\text{grounding}} \in [0.0, 1.0]$.
* **Implementación:**
  * Función: `chimera_metrics.task1.compute_section_grounding`
  * Clase: `chimera_metrics.task1.SectionGroundingMetric`
  * Referencia oficial: `evaluate.py::section_grounding_score` (L940)

---

## 4. Tarea 3: Pronóstico y Análisis de Supervivencia Post-Quirúrgica

### 4.1 Coincidencia de Evento de Recurrencia Bioquímica (BCR)
* **Ecuación:**
  ```
  S_event(e_hat, e) = 1.0  si e_hat == e,  0.0  si e_hat != e
  ```
  donde $e, \hat{e} \in \{0, 1\}$.
* **Implementación:**
  * Función: `chimera_metrics.task3.compute_recurrence_event_score`
  * Referencia oficial: `evaluate.py::recurrence_event_score` (L784)

### 4.2 Error Temporal Consciente de Censura (*Censoring-Aware Time Score*)
Mide la exactitud en la estimación del tiempo hasta la recurrencia o último seguimiento.
* **Ecuación:**
  ```
  Scale = max(T_gt, 1.0)

  Si e_gt == 1 (evento observado):
      S_time = max( 0.0, 1.0 - |T_hat - T_gt| / Scale )

  Si e_gt == 0 (paciente censurado a T_gt):
      S_time = 1.0                                 si T_hat >= T_gt
      S_time = max( 0.0, 1.0 - (T_gt - T_hat) / Scale )  si T_hat < T_gt
  ```
  donde $T_{\text{gt}}, \hat{T}$ representan los meses hasta la recurrencia o seguimiento. Rango: $S_{\text{time}} \in [0.0, 1.0]$.
* **Implementación:**
  * Función: `chimera_metrics.task3.compute_recurrence_time_score`
  * Clase: `chimera_metrics.task3.RecurrenceCaseScoreMetric`
  * Referencia oficial: `evaluate.py::recurrence_time_score` (L792)

### 4.3 Índice de Concordancia de Harrell ($C\text{-index}$)
Mide la capacidad de discriminar y ordenar correctamente los riesgos de falla en pares de pacientes.
* **Ecuación:**
  ```
  C = sum_{i, j: T_i < T_j, E_i = 1} [ I(T_hat_i < T_hat_j) + 0.5 * I(T_hat_i == T_hat_j) ] / sum_{i, j: T_i < T_j, E_i = 1} 1
  ```
  donde el par $(i, j)$ es admisible si el sujeto $i$ experimentó el evento antes del tiempo $T_j$. Rango: $C \in [0.0, 1.0]$.
* **Implementación:**
  * Función: `chimera_metrics.task3.compute_concordance_index`
  * Clase: `chimera_metrics.task3.HarrellCIndexMetric`
  * Referencia oficial: `evaluate.py::concordance_index` (L814)

### 4.4 Área Dinámica Acumulada Dependiente del Tiempo ($\text{AUC}(t^*)$ con IPCW)
Estima la discriminación temporal en horizontes fijos $t^* \in \{12, 24, 36, 60\}$ meses utilizando la probabilidad inversa de censura ponderada (IPCW de Uno et al.).
* **Ecuación:**
  ```
  AUC(t*) = sum_{i in Casos, j in Controles} [ I(eta_i > eta_j) + 0.5 * I(eta_i == eta_j) ] * W_i(t*) * W_j(t*) / [ sum_{i} W_i(t*) * sum_{j} W_j(t*) ]
  ```
  donde $\eta = -\hat{T}$ es el score de riesgo predicho, y las ponderaciones con respecto a la supervivencia de censura Kaplan-Meier $\hat{G}(t) = P(C > t)$ son:
  * Casos ($T_i \le t^*, E_i = 1$): $W_i(t^*) = 1.0 / \hat{G}(T_i)$
  * Controles ($T_j > t^*$): $W_j(t^*) = 1.0 / \hat{G}(t^*)$
* **Implementación:**
  * Funciones: `chimera_metrics.task3.compute_censoring_km`, `chimera_metrics.task3.compute_time_dependent_auc`
  * Clase: `chimera_metrics.task3.TimeDependentAUCMetric`
  * Referencia oficial: `evaluate.py::time_dependent_auc` (L880)

---

## 5. Racionalidad Clínica Cualitativa (LLM-as-a-Judge)

* **Rúbrica de Decisión (Tareas 1 y 2):** Evaluada con G-Eval midiendo alineación diagnóstica, mención de variables críticas, fidelidad factual y calibración de certeza.
* **Rúbrica de Pronóstico (Tarea 3):** Evaluada con G-Eval midiendo consistencia temporal interna y citas de hallazgos histopatológicos post-quirúrgicos (márgenes quirúrgicos, invasión a vesículas seminales, estadio patológico pT, ISUP).
* **Implementación:**
  * Clase: `chimera_metrics.judge.LLMRationaleJudge`
  * Rúbricas: `chimera_metrics.judge.DECISION_RATIONALE_RUBRIC`, `chimera_metrics.judge.RECURRENCE_RATIONALE_RUBRIC`
  * Referencia oficial: `evaluate.py::build_rationale_judge` (L1225)
