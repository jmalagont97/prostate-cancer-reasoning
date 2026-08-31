# Experiment Design: MCCV-30 estratificado para las tres subtareas de Task 2
**Experiment**: `experiments/exp_15/`  
**Project**: `prostate-cancer-reasoning`  
**Date**: 2026-08-30  
**Author**: Juan Sebastian Malagón Torres  
**Status**: Draft

---

## 1. Hypothesis

La evaluación conjunta mediante MCCV-30, con estratificación simultánea de decisión terapéutica, confianza clínica y pesos de variables, permitirá medir de forma equilibrada las tres subtareas sin favorecer ninguna de ellas por la distribución de los datos.

## 2. Experimental Setup

- **Datos**: `data/processed/input.csv` y `data/processed/ground_truth.csv`, 70 casos con `pid` alineado. Antes de ejecutar este diseño, `ground_truth.csv` deberá ampliarse con las 11 etiquetas de `variable_weights` extraídas de los JSON de razonamiento; el archivo actual solo contiene decisión y confianza.
- **Entradas**: variables clínicas tabulares, embeddings MRI y narrativa clínica; la modalidad WSI no forma parte de este benchmark y se evaluará posteriormente como extensión.
- **Subtareas**:
  1. `treatment_decision`: `active_treatment`, `active_surveillance`, `continued_surveillance`, `watchful_waiting`.
  2. `confidence`: `clear`, `borderline`, `uncertain`.
  3. `variable_weights`: nivel ordinal por variable clínica, evaluado sobre las 11 variables oficiales.
- **Validación**: 30 particiones Monte Carlo cross-validation, con semilla maestra fija y semillas derivadas registradas por partición.
- **Estratificación conjunta**: cada partición deberá preservar, dentro de la variación muestral esperable, las distribuciones de las tres subtareas. Se usará una representación multilabel/ordinal de cada caso: decisión, confianza y los 11 niveles de `variable_weights`; las combinaciones con frecuencia insuficiente se agruparán determinísticamente antes de estratificar, sin usar información del conjunto de validación para ajustar transformaciones. Si las etiquetas de pesos no están disponibles, la campaña quedará bloqueada y no se sustituirá por estratificación solo de decisión.
- **Regla de fuga**: toda normalización, selección de variables, TF-IDF, calibración y ajuste de hiperparámetros se aprenderá exclusivamente en el entrenamiento de cada partición.

## 3. Unidad experimental

El diseño no fija ningún modelo, arquitectura, algoritmo ni hiperparámetro. Cada modelo que se evalúe posteriormente deberá ejecutarse sobre exactamente los mismos 30 splits y bajo el mismo protocolo de preprocesamiento, predicción y evaluación. La selección de modelos pertenece a experimentos de implementación posteriores y no forma parte de este diseño metodológico.

## 4. Evaluation Protocol (chimera_metrics)

Queda estrictamente prohibido implementar funciones de evaluación ad-hoc o redundantes en los scripts experimentales. Toda evaluación debe consumir directamente el paquete canónico `utils/pkg/chimera_metrics` (`chimera_metrics.task2` y `chimera_metrics.core`).

### 4.1 Métricas canónicas por subtarea

1. **Subtarea 1 (Decisión terapéutica)**:
   - Exact Match Decision Gate: `compute_treatment_decision_gate` / `TreatmentDecisionGateMetric` ($S_{gate} \in \{0, 1\}$).
   - Macro-F1, Micro-F1, Weighted-F1 y F1 por clase calculados sobre las decisiones normalizadas mediante `norm_treatment_decision`.
   - Multiclass Brier Score y Matriz de Confusión.
2. **Subtarea 2 (Confianza clínica)**:
   - `compute_confidence_score` (penalización por distancia ordinal normalizada en escala $[0, 1]$).
   - Macro-F1 y Brier Score sobre clases `clear`, `borderline`, `uncertain`.
3. **Subtarea 3 (Pesos de variables clínicas y factores decisivos)**:
   - `compute_variable_weights_score` (distancia Manhattan ordinal normalizada sobre las 11 variables).
   - `compute_important_decisive_f1` / `extract_important_decisive_set` (F1 set-based sobre el conjunto de variables marcadas como `important` o `decisive`).
   - `Task2ReasoningMetric` como evaluador compuesto de razonamiento determinista.
4. **Métricas de Grounding y Eficiencia (cuando aplique)**:
   - `compute_section_grounding` y `compute_tool_efficiency`.
5. **Métrica integrada (Case Score)**:
   - Compuerta exacta $S_{gate}$: si $S_{gate} = 0 \implies \text{Case Score} = 0$. Si $S_{gate} = 1$, agregación lineal ponderada de los componentes calculados exclusivamente por `chimera_metrics`.

### 4.2 Resumen estadístico

Para cada configuración evaluada se guardarán resultados por partición, media, desviación estándar y su intervalo de confianza bootstrap al 95 %. Las comparaciones serán pareadas por partición fija y reportadas de forma simétrica sin suposiciones de un baseline particular.

### 4.3 Artefactos

Cada condición deberá conservar sus predicciones por `pid` y partición, métricas por subtarea, matriz de confusión, calibración, composición de estratos, versión del código y configuración exacta. Los resultados se almacenarán en `experiments/exp_15/results/<condition>/`.

## 5. Decision Rules

1. **Criterio global**: mayor media de `Case Score` en validación cruzada pareada, evaluada de forma conjunta y sin ponderaciones unilaterales entre las tres subtareas.
2. **Decisión terapéutica**: evaluación rigurosa mediante Macro-F1 y F1 desagregado por clase, garantizando la estabilidad frente al desbalance.
3. **Confianza**: evaluación mediante Macro-F1 y Brier; cualquier clase sin observaciones se clasifica como no evaluable.
4. **Pesos de variables**: evaluación mediante MAE ordinal y concordancia exacta en los niveles críticos (`important` / `decisive`).
5. **Consistencia multimodal**: ningún enfoque se considerará aceptable si optimiza una sola subtarea a costa de una degradación severa en las otras dos.

## 6. Comparabilidad y análisis controlado

Todos los modelos o configuraciones que se incorporen posteriormente deberán evaluarse sobre los mismos splits MCCV-30, con las mismas etiquetas, métricas y reglas de decisión. Las comparaciones de modalidades, arquitecturas o hiperparámetros se documentarán en diseños experimentales específicos, no en este protocolo transversal.

La estratificación solo por decisión podrá utilizarse únicamente como análisis metodológico de sensibilidad frente a la condición canónica de estratificación conjunta; no será el protocolo principal.

## 7. Risks and Mitigations

- **Inconsistencia y fragmentación evaluativa**: implementación de métricas ad-hoc no oficiales. *Mitigación*: **Uso exclusivo y mandatorio de `utils/pkg/chimera_metrics`** (`chimera_metrics.task2` y `chimera_metrics.core`). Ningún script experimental implementará lógica de métricas propia.
- **Desbalance extremo**: `watchful_waiting` tiene solo 2 casos. Se reportará F1 por clase, intervalos amplios y estabilidad de aparición por split; no se afirmará superioridad estadística para esa clase.
- **Clase `uncertain` ausente**: no se imputará artificialmente. Se separará "no observable" de "rendimiento cero".
- **Sobreajuste por tamaño muestral**: se limitará la búsqueda de hiperparámetros y se usarán comparaciones pareadas sobre splits fijos.
- **Métrica compuesta opaca**: se publicarán todos los componentes del `Case Score`, además de la métrica agregada calculada por `chimera_metrics`.
- **WSI ausente**: este diseño no permite concluir sobre el sistema multimodal completo del reto; la incorporación de biopsia/WSI requiere una extensión planificada y una nueva evaluación.
- **Ground truth incompleto**: el `ground_truth.csv` actual no contiene `variable_weights`; se debe generar esa salida antes de construir los splits MCCV-30.

## 8. Reproducibility Checklist

- [ ] MCCV-30 y semillas derivadas registradas.
- [ ] Estratos conjuntos y tamaños por partición guardados.
- [ ] Transformaciones ajustadas únicamente en entrenamiento.
- [ ] Predicciones por caso y partición almacenadas.
- [ ] Configuración y versión del código registradas en cada condición.
- [ ] Métricas oficiales y métricas por subtarea calculadas con la misma versión de `chimera_metrics`.
- [ ] Estado de Git documentado; actualmente existen cambios locales no comprometidos.

## 9. Next Steps

1. Extraer y validar las 11 etiquetas de `variable_weights` en `ground_truth.csv`.
2. Revisar y aprobar este diseño experimental.
3. Tras la aprobación, elaborar un `IMPLEMENTATION.md` para `exp_15` con los scripts, configuraciones y utilidades concretas.
4. Ejecutar la validación de los splits y la auditoría de estratificación sobre las tres subtareas.
5. Incorporar WSI en una extensión independiente después de validar este benchmark.
