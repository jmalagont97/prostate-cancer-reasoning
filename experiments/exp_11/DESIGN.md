# Experiment Design: Task 3 Raw-Data Preprocessing and Multimodal Reconciliation

**Experiment**: `experiments/exp_11/`  
**Project**: `prostate-cancer-reasoning`  
**Date**: 2026-08-29  
**Author**: Juan Sebastian Malagón Torres  
**Status**: Draft

---

## 1. Hypothesis

Es posible transformar de forma determinista los archivos crudos de CHIMERA-Agent Task 3 en una representación multimodal reproducible, preservando todos los `case_id`, las variables clínicas y patológicas, las etiquetas de supervivencia y los embeddings, sin fuga de información ni pérdida silenciosa de casos.

Este experimento evalúa la fidelidad y calidad del preprocesamiento; no pretende demostrar todavía una mejora del modelo predictivo.

## 2. Experimental Setup

- **Tarea**: CHIMERA-Agent Task 3, predicción del tiempo hasta recurrencia bioquímica después de prostatectomía radical.
- **Datos crudos**: `../../data/raw/train_release/task3/`. La ruta solicitada originalmente como `train_rlease` no existe; `train_release` es la ruta verificada.
- **Unidad de análisis**: un caso identificado por `case_id`.
- **Archivos esperados por caso**:
  - `structured-prompt.json`;
  - `prostate-time-to-recurrence-or-last-follow-up-clinical-data.json`;
  - `prostate-modality-level-neural-representations.json`;
  - `prostate-time-to-recurrence-or-last-follow-up.json`.
- **Población de referencia conocida**: 75 casos, con 19 recurrencias observadas y 56 observaciones censuradas, según la propuesta de `origin/rdacostav_dev`; el script deberá calcular y reportar estos valores, no asumirlos silenciosamente.
- **Referencia metodológica**: propuesta `origin/rdacostav_dev/task3_solution/`, especialmente la extracción estructurada de patología, la alineación explícita por `case_id` y el tratamiento de las bolsas de embeddings.
- **Resultado previsto**: representación intermedia tabular y multimodal, junto con un manifiesto de cobertura, dimensiones, cardinalidades, ausencias y errores.

## 3. File Layout for This Experiment

```text
experiments/exp_11/
├── DESIGN.md                  <- este diseño experimental
├── IMPLEMENTATION.md          <- se creará después de aprobar el diseño
├── scripts/                   <- script y configuración, definidos después
├── results/                   <- manifiestos y salidas del procesamiento
└── reports/                   <- informe de validación del preprocesamiento
```

El script deberá leer únicamente desde `data/raw/train_release/task3/` y escribir la representación procesada canónica en `data/processed/task3/`, sin modificar los datos crudos. Los manifiestos, logs y resultados de validación del experimento permanecerán bajo `experiments/exp_11/`.

## 4. Baselines / Reference Conditions

| Condición | Descripción | Propósito |
|---|---|---|
| `raw_inventory` | Inventario de casos, archivos y modalidades existentes | Establecer el universo de casos y detectar archivos faltantes |
| `structured_extraction` | Extracción de edad, PSA, DRE y tratamiento previo | Validar los campos del prompt estructurado |
| `pathology_extraction` | Extracción de variables desde `surgical_pathology_report` | Validar cobertura y semántica de patología posquirúrgica |
| `survival_labels` | Lectura separada de `months_to_recurrence` y `event` | Evitar confundir censura, evento y ausencia de etiqueta |
| `embedding_extraction` | Lectura por `case_id` de MRI, biopsia y prostatectomía | Validar dimensiones, bolsas y alineación |
| `full_preprocessing` | Ejecución conjunta de todas las condiciones anteriores | Generar la representación final reconciliada |

## 5. Proposed Processing Conditions

### 5.1 Campos estructurados

Extraer y conservar los valores originales necesarios para auditoría, además de sus representaciones numéricas cuando sea posible:

```text
age
psa
dre
active_treatment_prior_to_surgery
```

El texto DRE se debe mapear a:

```text
dre_abnormal
dre_bilateral
dre_nodule
dre_ct_stage_ordinal
```

Los valores no observados deben permanecer como faltantes. No se deben transformar automáticamente en negativos.

### 5.2 Patología quirúrgica

Extraer desde `surgical_pathology_report` las variables utilizadas en la propuesta de `rdacostav_dev`:

```text
isup_grade_group
gleason_primary
gleason_secondary
gleason_sum
gleason_tertiary_present
pt_stage_ordinal
epe_present
margins_positive
svi_present
lvi_present
ln_assessed
ln_positive
```

Los patrones de extracción deben validarse contra las frases observadas en el corpus y registrar los casos que no coincidan. No se deben inventar categorías clínicas para frases desconocidas.

La condición ganglionar debe preservar la diferencia entre:

```text
ln_assessed = 0, ln_positive = missing
ln_assessed = 1, ln_positive = 0
ln_assessed = 1, ln_positive = 1
```

### 5.3 Etiquetas de supervivencia

Conservar en un bloque separado:

```text
months_to_recurrence
 event
```

`event` debe permanecer binario y separado de las características de entrada. Ninguna variable derivada de la etiqueta puede entrar en la matriz de predictores.

### 5.4 Embeddings multimodales

Leer y alinear por `case_id`:

```text
MRI image                  -> vector de dimensión 1024
Biopsy slide               -> bolsa de 0–3 vectores de dimensión 960
Prostatectomy slide        -> bolsa de 0–3 vectores de dimensión 960
```

El preprocesamiento debe conservar las bolsas y su cardinalidad real. El masked mean pooling y la PCA pertenecen al paso de modelado posterior: no se deben ajustar durante este experimento ni sobre todo el conjunto antes de la validación cruzada.

### 5.5 Reconciliación final

La salida final debe mantener un registro por caso y separar claramente:

```text
case_id
features clínicas/patológicas
labels de supervivencia
embeddings o referencias a embeddings
estado de validación
```

El proceso debe ser aditivo y no alterar los archivos crudos.

## 6. Evaluation Protocol

Se evaluará el preprocesamiento mediante controles de integridad y cobertura:

1. Reconciliación exacta del conjunto de `case_id` entre todos los archivos disponibles.
2. Conteo de casos completos, incompletos y con archivos ausentes.
3. Cobertura por campo extraído y número de coincidencias/no coincidencias de cada patrón.
4. Conteo de eventos y censuras a partir de las etiquetas reales.
5. Verificación de dimensiones de embeddings y cardinalidad por modalidad.
6. Detección de duplicados, colisiones de `case_id` y alineación accidental por orden de filas.
7. Verificación de que ninguna etiqueta de supervivencia aparece en las características.
8. Segunda ejecución idéntica para comprobar determinismo.
9. Validación de que un caso con información ausente queda explícitamente registrado y no desaparece silenciosamente.

Los artefactos se guardarán en `results/<condition>/`, incluyendo un manifiesto de casos y un resumen de validación. Cada ejecución debe registrar la versión exacta del código y el estado de Git.

Este experimento no reportará C-index como métrica primaria. El C-index se evaluará en un experimento posterior, cuando el modelo Cox, la imputación y la PCA se ajusten dentro de cada fold de entrenamiento.

## 7. Expected Results & Decision Rules

### Criterios de aceptación

El preprocesamiento se considerará válido si:

- todos los casos detectados en el inventario aparecen en la salida o quedan explicados en el manifiesto de errores;
- los `case_id` se alinean sin depender del orden de los archivos;
- las etiquetas `months_to_recurrence` y `event` se conservan sin alteración;
- las dimensiones y bolsas de embeddings son válidas;
- los patrones clínicos y patológicos producen cobertura reproducible;
- las ausencias clínicas se representan como faltantes explícitos;
- no se incorpora información de recurrencia a las características;
- dos ejecuciones con la misma entrada producen resultados equivalentes.

### Decisiones posteriores

- **Si se cumplen todos los criterios**: aceptar la representación para el entrenamiento de Task 3 y preparar un plan de implementación separado.
- **Si hay casos faltantes o patrones no reconocidos**: detener la promoción, auditar esos casos y corregir el mapeo antes de entrenar.
- **Si hay discrepancias de etiquetas o dimensiones**: considerar el experimento fallido hasta resolver la causa.
- **Si la cobertura es parcial pero clínicamente interpretable**: conservar el caso con faltantes explícitos y documentar la decisión; no eliminarlo silenciosamente.

## 8. Risks & Mitigations

- **Ruta de datos incorrecta**: usar la ruta verificada `data/raw/train_release/task3/` y registrar la discrepancia con `train_rlease`.
- **Fuga de etiquetas**: separar etiquetas de supervivencia de las características y verificar los nombres de columnas antes del modelado.
- **Pérdida silenciosa de casos**: usar unión por `case_id`, manifiesto de casos y reconciliación exacta de conteos.
- **Regex incompletos o ambiguos**: registrar frases no reconocidas y validar patrones contra el corpus completo.
- **Confusión entre ganglios no evaluados y negativos**: conservar `ln_assessed` y `ln_positive` como estados distintos.
- **Ajuste prematuro de PCA o imputación**: limitar este experimento a extracción y organización; ajustar transformaciones estadísticas dentro de cada fold en el experimento de modelado.
- **Inconsistencia con el contrato de Task 3**: conservar explícitamente `event` en las etiquetas y verificar la salida oficial antes del agente de inferencia.

## 9. Reproducibility Checklist

- [ ] Ruta exacta y versión del dataset registradas.
- [ ] Conteo y manifiesto de `case_id` guardados.
- [ ] Patrones de extracción y cobertura por campo registrados.
- [ ] Dimensiones y cardinalidades de embeddings verificadas.
- [ ] Etiquetas y características almacenadas por separado.
- [ ] Ejecución determinista comprobada.
- [ ] Estado de Git y versión del código registrados por ejecución.
- [ ] Datos crudos no modificados.
- [ ] Resultados guardados bajo `experiments/exp_11/results/`.
- [ ] Informe de validación guardado bajo `experiments/exp_11/reports/`.

## 10. Next Steps

1. Aprobar este diseño experimental.
2. Elaborar y aprobar `experiments/exp_11/IMPLEMENTATION.md`, que definirá el script, configuración y artefactos concretos.
3. Implementar el preprocesamiento en `jmalagont_dev`.
4. Ejecutar las condiciones de validación sobre `data/raw/train_release/task3/`.
5. Revisar los resultados antes de conectar la representación con el modelo de supervivencia.
