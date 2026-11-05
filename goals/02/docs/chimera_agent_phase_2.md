# CHIMERA-Agent Phase 2: Challenge Specification & Evaluation Benchmark

**Documento:** Especificación Técnica Completa del Reto CHIMERA-Agent (Fase 2)  
**Fuentes Oficiales:** 
- [Grand Challenge — CHIMERA-Agent Phase 2](https://chimera-agent.grand-challenge.org/chimera-agent-phase-2/)
- [Zenodo Training Release v5 (DOI: 10.5281/zenodo.23886587)](https://zenodo.org/records/23886587)  
**Línea de Investigación Activa:** `goals/02/`  
**Última Actualización:** 2026-11-05  

---

## 1. Visión General del Reto

El benchmark **CHIMERA-Agent** (*Combining HIstology, Medical imaging and molEcular data for medical PRognosis and diAgnosis Agent*) plantea la construcción de agentes de inteligencia artificial médica multimodal capaces de navegar y resolver la trayectoria clínica completa de pacientes con sospecha o diagnóstico de cáncer de próstata (PCa). 

El reto reproduce de forma fidedigna las complejidades del entorno hospitalario real:
- **Disponibilidad asíncrona de información:** La evidencia diagnóstica no se encuentra disponible simultáneamente; se genera en etapas clínicas secuenciales donde el agente debe solicitar activamente nuevos estudios o reportes.
- **Fragmentación y valores faltantes:** Estudios omitidos, historiales incompletos y modalidades ausentes de forma deliberada.
- **Conflicto intermodal:** Discrepancias entre biomarcadores séricos (PSA), hallazgos radiológicos (mpMRI / PI-RADS), modelos de deep learning de soporte y confirmación histopatológica (Gleason / ISUP).

---

## 2. Novedades Oficiales de Fase 2 (Especificación Zenodo v5 / Grand Challenge)

A diferencia de la Fase 1, la **Fase 2** introduce una reestructuración profunda de los datos de entrada y de la comunicación agéntica para penalizar soluciones basadas en reglas estáticas y forzar **razonamiento interactivo dinámico**.

### 2.1 Metadatos del Dataset de Entrenamiento (Zenodo Release)
- **Registro Zenodo:** [DOI: 10.5281/zenodo.23886587](https://doi.org/10.5281/zenodo.23886587)
- **Versión:** Versión 5 (Publicada el 02 de Noviembre de 2026).
- **Autoría:** Nadieh Khalili et al. (Radboud University Medical Center).
- **Licencia:** *Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)*.
- **Acceso:** Archivo de datos restringido en Zenodo; distribuido a participantes verificados a través de la plataforma Grand Challenge.

---

### 2.2 Reorganización Estricta de Inputs: Qué se Eliminó del Prompt Estructurado

En la Fase 2, el archivo `structured-prompt.json` **ha sido despojado de la mayoría de variables clínicas críticas**. El agente ya no puede operar como un clasificador tabular ciego; debe invocar herramientas para inspeccionar los reportes del EHR.

#### Comparativa de Campos en `structured-prompt.json`:

| Tarea | Variables Eliminadas del `structured-prompt.json` en Fase 2 | Variables que Permanecen en `structured-prompt.json` |
|---|---|---|
| **Task 1** | `psa`, `psad`, `vol`, `pirads`, `psap`, `task`, `cspca`, `notes` | `case_id`, `age`, `dre`, `ct`, `bx`, `pmhx` (comorbilidades) |
| **Task 2** | `psa`, `psad`, `vol`, `pirads`, `psap`, `task`, `cspca`, `notes`, `bx_isup`, `bx_gl_prim`, `bx_gl_sec` | `case_id`, `age`, `dre`, `ct`, `bx`, `pmhx` |
| **Task 3** | `task` | `case_id`, `age`, `psa` (prequirúrgico), `dre`, `active_treatment_prior_to_surgery` |

> [!WARNING]
> Obsérvese que incluso el valor basal de PSA (`psa`), el PSA previo (`psap`), el volumen (`vol`), PI-RADS (`pirads`) y las notas narrativas (`notes`) fueron **completamente eliminados del prompt estructurado en las Tareas 1 y 2**. El agente debe consultar explícitamente el reporte de radiología y el panel de laboratorio/tendencia de PSA para recuperarlos.

---

### 2.3 Desacoplamiento de Predicciones de IA en `*-clinical-data.json`

En la Fase 1, las predicciones generadas por modelos de deep learning estaban incrustadas en el texto libre de los reportes médicos. En la Fase 2, **se desacoplan como campos independientes en el archivo de datos clínicos**:

1. **Probabilidad csPCa por Deep Learning (`cspca`):**
   - Corresponde a la estimación del modelo PI-CAI de cáncer de próstata clínicamente significativo.
   - En Fase 2, se entrega como un campo raíz independiente en `*-clinical-data.json` y **ya no está incluido dentro del texto de `radiology_report`**.
   - *Nota técnica:* Es una probabilidad no calibrada entre $0$ y $1$; no equivale a una puntuación de riesgo formal.
2. **Grado ISUP Predicho por IA (`AI_predicted_isup_grade_group`):**
   - En Tareas 2 y 3, se proporciona como un campo raíz independiente en `*-clinical-data.json` y **ya no está incluido dentro de `pathology_report`**.
   - *Regla canónica de Zenodo:* Corresponde estrictamente a la biopsia más reciente disponible del paciente. Si el paciente no cuenta con predicción de IA o biopsia previa, el valor es explícitamente `null`.

---

### 2.4 Interfaz de Herramientas MCP Obligatoria (Model Context Protocol)

- Los participantes deben interactuar con el entorno hospitalario utilizando la **interfaz MCP oficial** proporcionada por el reto CHIMERA-Agent.
- Quedan prohibidas interfaces de herramientas propietarias o scripts de scraping ad-hoc sobre el sistema de archivos del contenedor de evaluación.

---

## 3. Estructura de Archivos por Caso y Disponibilidad de Etiquetas

Para cada paciente, los datos de entrada se organizan en una carpeta individual bajo tres posibles tareas:

```text
dataset/
├── task1/
│   └── <patient_folder>/
│       ├── structured-prompt.json
│       ├── prostate-biopsy-decision-clinical-data.json
│       ├── prostate-modality-level-neural-representations.json
│       ├── prostate-biopsy-decision.json             (Ground Truth — 91 casos)
│       └── prostate-biopsy-decision-reasoning.json   (Ground Truth — 91 casos)
├── task2/
│   └── <patient_folder>/
│       ├── structured-prompt.json
│       ├── prostate-treatment-decision-clinical-data.json
│       ├── prostate-modality-level-neural-representations.json
│       ├── prostate-treatment-decision.json           (Ground Truth — 72 casos)
│       └── prostate-treatment-decision-reasoning.json (Ground Truth — 72 casos)
└── task3/
    └── <patient_folder>/
        ├── structured-prompt.json
        ├── prostate-time-to-recurrence-or-last-follow-up-clinical-data.json
        ├── prostate-modality-level-neural-representations.json
        └── prostate-time-to-recurrence-or-last-follow-up.json (Ground Truth — 75 casos)
```

### Estado de Disponibilidad de Ground Truth (Training Set Zenodo v5):

| Tarea | Total Carpetas Insumo | Casos con Etiqueta Decisión | Casos con Razonamiento de Referencia | Estado de `reveal_sequence` en Referencia |
|---|---|---|---|---|
| **Task 1** | 195 | 91 | 91 | **Disponible** (Lista de secciones consultadas por el patólogo) |
| **Task 2** | 153 | 72 | 72 | **No disponible temporalmente** (representado como lista vacía `[]`) |
| **Task 3** | 75 | 75 | 0 | Sin razonamiento de referencia en release de entrenamiento |

---

## 4. Representaciones Neuronales de Modelos Fundacionales

El archivo `prostate-modality-level-neural-representations.json` contiene vectores latentes densos extraídos por modelos fundacionales pre-entrenados:

```json
{
  "MRI image": [
    [ /* Vector denso de 1024 dimensiones */ ]
  ],
  "Biopsy slide": [
    [ /* Vector lámina 1 de 960 dimensiones */ ],
    [ /* Vector lámina 2 de 960 dimensiones */ ]
  ],
  "Prostatectomy slide": []
}
```

- **`"MRI image"`:** Vector $1 \times 1024$ dimensiones (representación multimodal de la secuencia axial T2w, mapa ADC y DWI de alto valor b). Siempre presente si hay RM.
- **`"Biopsy slide"`:** Lista de $0$ a $3$ vectores de $960$ dimensiones, donde cada vector representa una lámina histopatológica digitalizada (WSI H&E) de biopsia.
- **`"Prostatectomy slide"`:** Lista de $0$ a $3$ vectores de $960$ dimensiones de láminas quirúrgicas de prostatectomía.
- **Modalidades Faltantes:** Se representan obligatoriamente como listas vacías `[]`. Los agentes deben manejar de forma nativa la ausencia de láminas o imágenes.

---

## 5. Desglose Exhaustivo de las Tres Tareas Clínicas

### 5.1 Task 1: Decisión Diagnóstica de Biopsia (Pre-Histológica)
- **Pregunta:** ¿Se debe realizar biopsia de próstata al paciente?
- **Salida Decisión:** `prostate-biopsy-decision.json` con `"yes"` o `"no"`.
- **Salida Razonamiento:** `prostate-biopsy-decision-reasoning.json`
  - `confidence`: `"clear"` | `"borderline"` | `"uncertain"`.
  - `variable_weights`: Diccionario con 10 variables clínicas obligatorias (`age`, `fh`, `cspca`, `pirads`, `vol`, `psa`, `comorbidity`, `psad`, `dre`, `bx`), asignando uno de 4 niveles:
    - `"not_used"` (0)
    - `"noted"` (1)
    - `"important"` (2)
    - `"decisive"` (3)
  - `reveal_sequence`: Lista de nombres de secciones clínicas/herramientas consultadas (e.g., `["radiology_report", "laboratory_results", "psa_trend"]`).
  - `free_text`: Explicación en lenguaje natural de los factores determinantes.

---

### 5.2 Task 2: Estratificación de Riesgo y Decisión Terapéutica Post-Biopsia
- **Pregunta:** ¿Cuál es la estrategia de manejo clínico óptima tras confirmar la biopsia?
- **Salida Decisión:** `prostate-treatment-decision.json` con exactamente una de las cuatro categorías:
  - `"active_surveillance"` (Vigilancia activa)
  - `"active_treatment"` (Tratamiento activo: prostatectomía, radioterapia)
  - `"continued_surveillance"` (Vigilancia continuada)
  - `"watchful_waiting"` (Espera vigilante para pacientes añosos/comórbidos)
- **Salida Razonamiento:** Idéntico esquema que Task 1, pero evaluando 11 variables clínicas:
  - `bx_isup`, `bx_gl_prim`, `bx_gl_sec`, `pirads`, `psad`, `psa`, `ct`, `cspca`, `age`, `fh`, `comorbidity`.

---

### 5.3 Task 3: Pronóstico y Tiempo hasta Recidiva Bioquímica Post-Prostatectomía
- **Pregunta:** Tras la prostatectomía radical, ¿cuándo se presentará la recidiva bioquímica (BCR)?
- **Salida Evento/Tiempo:** `prostate-time-to-recurrence-or-last-follow-up.json`:
  ```json
  {
    "months_to_recurrence": 48.5,
    "event": 1
  }
  ```
  - `event`: Entero binario (`1` = evento de recidiva confirmado, `0` = observación censurada).
  - `months_to_recurrence`: Flotante continuo que representa el tiempo en meses.
- **Salida Razonamiento:** `prostate-time-to-recurrence-or-last-follow-up-reasoning.json` conteniendo `free_text` con la justificación oncológica.

---

## 6. Sistema de Evaluación y Reglas de Puntuación

### 6.1 Puntuación General del Leaderboard

$$\text{Overall Score} = 0.4 \times \text{Score}_{\text{Task 1}} + 0.4 \times \text{Score}_{\text{Task 2}} + 0.2 \times \text{Score}_{\text{Task 3}}$$

1. **Task 1 Score:** Promedio entre el `Mean Case Score` y el `F1` de la clase positiva (`yes`):
   $$\text{Score}_{\text{Task 1}} = \frac{\overline{\text{case\_score}} + \text{F1}_{\text{biopsy (yes)}}}{2}$$
2. **Task 2 Score:** Promedio entre el `Mean Case Score` y el `Weighted F1` entre las cuatro clases terapéuticas:
   $$\text{Score}_{\text{Task 2}} = \frac{\overline{\text{case\_score}} + \text{F1}_{\text{weighted}}}{2}$$
3. **Task 3 Score:** Índice de Concordancia de Harrell (**Harrell's C-index**), evaluando la capacidad del modelo para ordenar el riesgo relativo entre pares de pacientes comparables respetando censura.

---

### 6.2 Decision Gate y Fórmula de `case_score` (Tareas 1 y 2)

El evaluador oficial ejecuta un **Decision Gate estricto**:
- Si la decisión primaria es incorrecta (`biopsy_decision` erróneo en T1 o `treatment_recommendation` erróneo en T2):
  $$\mathbf{case\_score = 0.0}$$
- Si la decisión primaria es acertada, se evalúan 6 componentes ponderados:

$$\text{case\_score} = \sum_{k} w_k \cdot S_k$$

| Métrica Componente | Peso (con Rationale LLM) | Peso (sin Rationale LLM) | Fundamento Evaluativo |
|---|---|---|---|
| **Confianza (`confidence_score`)** | 0.200 | 0.225 | Penalización por discrepancia ordinal con la confianza de referencia (`clear`, `borderline`, `uncertain`). |
| **Pesos de Variables (`variable_weight_score`)** | 0.250 | 0.275 | Error absoluto medio normalizado sobre la matriz de 10 u 11 variables ponderadas ($0$ a $3$). |
| **Factores Decisivos (`important_decisive_factor_score`)** | 0.150 | 0.175 | F1-score en la identificación exacta de variables etiquetadas como `important` o `decisive`. |
| **Uso de Herramientas (`tool_score`)** | 0.150 | 0.150 | Eficiencia y concordancia de las secciones solicitadas en `reveal_sequence` respecto a las requeridas. |
| **Anclaje de Secciones (`section_grounding_score`)** | 0.150 | 0.175 | Penaliza variables a las que el agente asignó peso positivo si la sección primaria que las contiene no fue consultada formalmente en `reveal_sequence` (según `section_variable_mapping.json`). |
| **Calidad de Rationale (`rationale_score`)** | 0.100 | 0.000 (re-normalizado) | Juicio semántico mediante LLM local offline (Gemma 4/3) sobre la coherencia y anclaje clínico de `free_text`. |

---

## 7. Mapeo Canónico de Variables a Secciones de Herramientas (`section_variable_mapping.json`)

Para no sufrir penalizaciones en el `section_grounding_score`, el agente debe conocer la fuente primaria de cada variable:

| Variable Clínica | Secciones Primarias en `reveal_sequence` | ¿Disponible Basalmente en Tarjeta? |
|---|---|---|
| `pirads` | `radiology_report` | No (Requiere consulta a reporte de RM) |
| `psad` | `radiology_report` | No (El volumen denominador se obtiene de RM) |
| `vol` | `radiology_report` | No (Volumen prostático de RM) |
| `cspca` | `radiology_report` / Tool `cspca` | No (Probabilidad deep learning) |
| `psa` | `psa_trend`, `laboratory_results` | Sí (Valor basal visible, pero trayectoria requiere trend/labs) |
| `age` | `comorbidity` (contexto funcional) | Sí (Edad visible directamente) |
| `dre` | `laboratory_results`, `previous_notes` | No (Requiere notas o examen clínico) |
| `fh` | `family_history`, `previous_notes` | No (Requiere antecedentes familiares) |
| `bx` / `bx_isup` / `bx_gl_*` | `pathology_report`, `previous_notes` | No (Requiere reporte patológico) |
| `comorbidity` | `previous_notes` | No (Requiere comorbilidades detalladas) |

---

## 8. Infraestructura, Entorno Docker y Restricciones Técnicas

1. **Entorno 100% Offline (Sin Conexión a Internet):**
   - Ejecución dentro de contenedores Docker con `--network none`.
   - Prohibido cualquier llamado a APIs externas comerciales (OpenAI, Anthropic, Gemini API, endpoints web).
   - Todos los modelos de inferencia (LLM locales, pesos de visión, embeddings) deben estar horneados en el contenedor.
2. **Límites de Cómputo en Grand Challenge:**
   - GPUs soportadas: **NVIDIA T4 (16 GB VRAM)** o **NVIDIA A10G (24 GB VRAM)**.
   - Memoria RAM: 16 o 32 GiB.
3. **Requisitos de Envío a Closed Testing Phase:**
   - **Manuscrito científico de 6 páginas** (plantilla MICCAI Springer LNCS, excluyendo referencias) con diagrama arquitectónico formal.
   - **Repositorio de código abierto en GitHub** bajo licencia permisiva (MIT, Apache 2.0, BSD).
   - Cuotas de envío: 5 envíos en fase de validación (75 casos), 1 envío final en test (250 casos con casos externos del Instituto Karolinska).
4. **Fechas Clave:**
   - **02 Nov 2026:** Lanzamiento de datos de entrenamiento Fase 2 (Zenodo v5).
   - **15 Nov 2026:** Apertura de validación en Grand Challenge.
   - **15 Dic 2026:** Cierre definitivo de envíos.

---

## 9. Implicaciones Estratégicas para el Enfoque Neurosimbólico en Goal 02

Los hallazgos del release Zenodo v5 reafirman y guían el diseño de **árboles de decisión neurosimbólicos inferidos desde el contexto**:

1. **Navegación Eficiente de Herramientas Guiada por Árboles:**
   - Dado que el prompt estructurado ya no incluye `psa`, `vol` ni `pirads`, el agente debe abrir la herramienta radiológica de inmediato.
   - Un árbol de decisión neurosimbólico determina qué nodos evaluar primero (e.g. consultar radiología $\to$ verificar $PIRADS$ y $PSAD$ $\to$ si es ambiguo, consultar laboratorio o $csPCa$). Esto optimiza simultáneamente el `tool_score` y el `section_grounding_score`.
2. **Inmunidad al Decision Gate:**
   - El árbol impone cortes deterministas en los predicados críticos ($PIRADS \ge 4$, $PSAD > 0.15$), asegurando superar el gate de decisión binaria ($ds = 1.0$) y previniendo que el `case_score` colapse a cero.
3. **Extracción Directa de Pesos de Variables:**
   - La estructura jerárquica del árbol asigna de forma natural:
     - Nodos de corte decisivos $\to$ `"decisive"`.
     - Nodos de bifurcación secundaria consultados $\to$ `"important"`.
     - Variables verificadas en el historial pero no determinantes $\to$ `"noted"`.
     - Variables no transitadas en la rama $\to$ `"not_used"`.
