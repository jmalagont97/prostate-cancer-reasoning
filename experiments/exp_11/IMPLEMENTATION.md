# Implementation Plan: Task 3 Raw-Data Preprocessing (`exp_11`)

**Experiment**: `experiments/exp_11/`  
**Project**: `prostate-cancer-reasoning`  
**Author**: Juan Sebastian Malagón Torres  
**Date**: 2026-08-29  
**Status**: Implemented and verified

---

## 1. Scope and objective

Implementar un único pipeline reproducible de preprocesamiento que lea los datos crudos de:

```text
data/raw/train_release/task3/
```

 y genere la representación canónica en:

```text
data/processed/task3/
```

El pipeline no entrenará modelos, no ajustará imputación, no calculará PCA y no modificará ningún archivo bajo `data/raw/`.

El código ejecutable del experimento vivirá en:

```text
experiments/exp_11/scripts/preprocess_task3.py
```

Los manifiestos, logs y validaciones de ejecución se conservarán bajo:

```text
experiments/exp_11/results/
```

## 2. Interface del script

El script tendrá una interfaz de línea de comandos con rutas configurables y valores por defecto explícitos:

```text
--input-root data/raw/train_release/task3
--output-root data/processed/task3
--report-root experiments/exp_11/results
```

Modos previstos:

```text
inventory       inventario y reconciliación de casos
extract         extracción y escritura de la representación canónica
validate        validación de artefactos ya generados
all             inventory + extract + validate
```

La ejecución debe ordenar los casos y las columnas de forma determinista. Los errores por caso se registrarán en el manifiesto y no se ocultarán mediante `try/except` silencioso.

## 3. Descubrimiento y alineación de casos

1. Descubrir directorios de caso bajo `input-root`.
2. Usar el nombre del directorio como `case_id` y validar que corresponda al identificador interno cuando esté disponible.
3. Inventariar la presencia de cada archivo esperado.
4. Alinear todas las fuentes por `case_id`, nunca por el orden de lectura.
5. Registrar casos duplicados, archivos faltantes, JSON inválido y claves inesperadas.
6. Mantener un registro por cada caso descubierto, incluso si no puede procesarse completamente.

El inventario debe producir un artefacto de auditoría bajo `experiments/exp_11/results/` y una copia resumida dentro de `data/processed/task3/`.

## 4. Extracción tabular

La extracción se dividirá conceptualmente en bloques independientes para permitir auditoría.

### 4.1 Prompt estructurado

Leer `structured-prompt.json` y extraer, conservando también el valor crudo cuando sea necesario:

```text
age
psa
dre
active_treatment_prior_to_surgery
```

Derivar las representaciones numéricas de DRE definidas en el diseño:

```text
dre_abnormal
dre_bilateral
dre_nodule
dre_ct_stage_ordinal
```

Los valores ausentes o no reconocidos se representarán como faltantes explícitos y se contabilizarán.

### 4.2 Datos clínicos y patología

Leer el archivo clínico de Task 3 y extraer de `surgical_pathology_report`:

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

Los extractores deben devolver tanto el valor normalizado como un estado de extracción:

```text
matched
missing
unrecognized
ambiguous
```

Las frases no reconocidas y las coincidencias múltiples se guardarán en un reporte de auditoría para revisión manual. No se asignará un valor clínico por defecto a una frase ambigua.

### 4.3 Etiquetas

Leer `prostate-time-to-recurrence-or-last-follow-up.json` en un bloque independiente y escribir:

```text
case_id
months_to_recurrence
event
```

Validaciones:

- `months_to_recurrence` debe ser numérico y no negativo;
- `event` debe estar en `{0, 1}`;
- las etiquetas no podrán entrar en el archivo de características;
- si falta una etiqueta, el caso queda marcado como incompleto y no se elimina silenciosamente.

## 5. Extracción de embeddings

Leer `prostate-modality-level-neural-representations.json` y preservar las modalidades sin aplicar pooling ni reducción de dimensionalidad:

```text
MRI image:                 vector de dimensión 1024
Biopsy slide:              matriz de 0–3 x 960
Prostatectomy slide:       matriz de 0–3 x 960
```

Se validará:

- que cada vector sea numérico;
- que las dimensiones sean constantes;
- que las bolsas vacías se conserven como bolsas vacías;
- que no se confunda una modalidad ausente con un vector de ceros;
- que el orden interno de las láminas se preserve;
- que cada bolsa quede asociada al `case_id` correcto.

El pooling promedio y la PCA se realizarán únicamente en el pipeline posterior de modelado y dentro de cada fold de entrenamiento.

## 6. Artefactos de salida

La representación canónica se escribirá en `data/processed/task3/` con los siguientes artefactos:

```text
data/processed/task3/
├── clinical_pathology_features.csv
├── survival_labels.csv
├── embeddings.npz
├── case_manifest.csv
└── schema.json
```

### `clinical_pathology_features.csv`

Una fila por `case_id`, ordenada determinísticamente, con las variables clínicas y de patología normalizadas. No contendrá `months_to_recurrence`, `event` ni ninguna variable derivada de ellas.

### `survival_labels.csv`

Una fila por `case_id` con:

```text
case_id
months_to_recurrence
event
```

### `embeddings.npz`

Contendrá matrices y arrays indexados por el orden documentado de `case_id`, incluyendo máscaras o longitudes para las bolsas de láminas. El índice de alineación se guardará explícitamente y no se inferirá de un orden externo.

### `case_manifest.csv`

Contendrá, por caso:

```text
case_id
all_expected_files_present
clinical_status
pathology_status
label_status
embedding_status
n_mri_vectors
n_biopsy_slides
n_prostatectomy_slides
error_count
```

### `schema.json`

Describirá las columnas, tipos, dimensiones, versión del formato y ruta de origen. No contendrá datos clínicos adicionales ni credenciales.

## 7. Validación y pruebas

El pipeline incluirá validaciones ejecutables para:

1. igualdad de los conjuntos de `case_id` entre tablas y el índice de embeddings;
2. ausencia de duplicados;
3. orden determinista de filas y columnas;
4. dimensiones `1024` para MRI y `960` para cada embedding de lámina;
5. cardinalidad real de cada bolsa;
6. dominio binario de `event`;
7. no negatividad y tipo numérico del tiempo;
8. separación entre características y etiquetas;
9. preservación explícita de faltantes;
10. conteo de casos completos, incompletos y con errores;
11. repetibilidad de dos ejecuciones sobre la misma entrada;
12. comprobación de que ningún archivo bajo `data/raw/` fue modificado.

Los resultados de validación se escribirán bajo:

```text
experiments/exp_11/results/full_preprocessing/
```

incluyendo `version.txt`, el manifiesto de ejecución, conteos, errores y un resumen JSON.

## 8. Reproducibilidad

Antes de cada ejecución se registrará:

```text
git commit completo
estado de git
ruta de entrada
ruta de salida
versión del esquema
fecha de ejecución
```

La ejecución no debe depender del directorio de trabajo actual. Todas las rutas por defecto se resolverán desde la raíz del repositorio o se recibirán explícitamente por CLI.

## 9. Criterios de aceptación

La implementación podrá ejecutarse cuando:

- el script esté escrito bajo `experiments/exp_11/scripts/`;
- los tests cubran inventario, extracción, faltantes, etiquetas y embeddings;
- la salida se genere exclusivamente en `data/processed/task3/`;
- exista alineación exacta por `case_id`;
- no haya pérdida silenciosa de casos;
- `event` y `months_to_recurrence` estén separados de las características;
- las dimensiones de embeddings sean verificadas;
- dos ejecuciones consecutivas produzcan resultados equivalentes;
- `git diff --check` y los tests relevantes terminen correctamente.

## 10. Secuencia de ejecución

1. Ejecutar `inventory` y revisar el manifiesto.
2. Ejecutar `extract` sobre todos los casos.
3. Ejecutar `validate` sobre los artefactos generados.
4. Repetir la ejecución en un directorio temporal de comparación para verificar determinismo.
5. Revisar manualmente las frases patológicas no reconocidas o ambiguas.
6. Generar el reporte de `exp_11` únicamente después de resolver o documentar las discrepancias.

## 10.1. Ejecución verificada

La implementación se ejecutó sobre `data/raw/train_release/task3/` y produjo la salida canónica en `data/processed/task3/`.

Resultados observados:

```text
casos procesados:       75
eventos:                19
casos censurados:       56
casos para revisión:     0
```

La segunda ejecución produjo el mismo hash agregado de los artefactos que la primera. La validación ejecutable terminó con `VALID` y no se detectaron modificaciones bajo `data/raw/`.

Los resultados de ejecución se conservaron en `experiments/exp_11/results/full_preprocessing/`.

## 11. Límites explícitos

Este plan no incluye:

- entrenamiento del modelo Cox;
- selección de `lambda`, `p` o número de componentes PCA;
- ajuste de PCA o imputación;
- generación de predicciones de Task 3;
- modificación de la salida oficial del agente;
- construcción de la imagen Docker de submission.

Esas actividades requieren un experimento o plan de implementación posterior y deben utilizar exclusivamente los artefactos procesados validados.
