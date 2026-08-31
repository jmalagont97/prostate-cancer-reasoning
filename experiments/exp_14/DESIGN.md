# Experiment Design: Validación end-to-end del agente Task 3 sobre la cohorte completa
**Experiment**: experiments/exp_14/  
**Project**: prostate-cancer-reasoning  
**Date**: 2026-08-30  
**Author**: Juan Sebastian Malagón Torres  
**Status**: Complete

---

## 1. Hypothesis

El agente Task 3 puede procesar de forma reproducible todos los casos disponibles, generar los dos archivos exigidos por el contrato y producir predicciones Cox válidas acompañadas de razonamiento clínico generado por Gemma 4 E2B, sin que Gemma altere `months_to_recurrence` ni `event`.

La evaluación cuantitativa sobre los 75 casos anotados será **exploratoria e in-sample**: el modelo productivo fue ajustado con esta misma cohorte, por lo que puede resultar optimista y no medirá generalización a casos de challenge no vistos.

## 2. Experimental Setup

- **Dataset**: todos los directorios de casos disponibles en `data/raw/train_release/task3/`; actualmente se esperan 75 casos (`T3-001`–`T3-072`, con identificadores faltantes según la distribución original).
- **Entrada del agente**: `structured-prompt.json` y los archivos clínicos/representaciones de cada caso, siguiendo el flujo Docker de `deploy/agent/`.
- **Agente**: imagen Docker Task 3 con Ollama y `gemma4:e2b` (versión inmutable), acelerada por GPU cuando esté disponible.
- **Orquestación**: invocación determinista de `predict_bcr_task3` → generación de razonamiento por Gemma; Gemma no puede modificar los valores Cox.
- **Salida esperada por caso**: JSON de resultado con `months_to_recurrence` y `event`, más JSON independiente de razonamiento.
- **Referencia anotada**: `prostate-time-to-recurrence-or-last-follow-up.json` de cada caso, usada solo después de completar las inferencias.
- **Repetibilidad**: una ejecución completa, con logs, estado/versionado y conteo de casos procesados; los casos fallidos no se excluyen silenciosamente.

## 3. File Layout for This Experiment

```
experiments/exp_14/
├── DESIGN.md
├── scripts/                 ← scripts/configuración definidos en IMPLEMENTATION.md
├── results/
│   ├── agent_all_cases/     ← outputs, logs, cobertura y validación de contrato
│   └── annotated_metrics/   ← métricas calculadas sobre casos anotados
└── reports/
    └── summary.md           ← generado posteriormente por experiment-reporter
```

Los nombres concretos de scripts y la estrategia de ejecución se definirán en el plan de implementación, no en este documento.

## 4. Baselines

| Referencia | Propósito | Métrica esperada |
|---|---|---|
| Predicción Cox productiva (`task3_cox_pca5.pkl`) | Referencia determinista del valor que debe preservar el agente | Igualdad exacta o tolerancia numérica documentada entre tool y output |
| Flujo agente Task 3 | Condición evaluada | Cobertura de contrato y métricas Task 3 |

No se comparará contra un modelo reentrenado dentro de este experimento; hacerlo mezclaría la validación del flujo con una nueva selección de modelo.

## 5. Proposed Conditions

| Condición | Casos | Evaluación |
|---|---:|---|
| `agent_all_cases` | Todos los casos disponibles, anotados y no anotados si aparecen en el directorio | Cobertura, errores, latencia, validez de ambos JSON y consistencia Cox→output |
| `annotated_metrics` | Subconjunto con ground truth válido | `event_score`, `time_score`, Harrell C-index y métricas Task 3 adicionales disponibles en `chimera_metrics` |

La segunda condición es una vista analítica de las salidas de la primera, no una segunda inferencia.

## 6. Evaluation Protocol

- **Cobertura primaria**: 100% de los casos descubiertos deben producir los dos archivos requeridos; cada error debe quedar registrado con su caso y causa.
- **Integridad del contrato**: `months_to_recurrence` numérico y finito; `event` binario; razonamiento presente como JSON independiente; ningún campo de razonamiento sustituye al outcome.
- **Consistencia de orquestación**: los valores finales deben coincidir con la respuesta de Cox; Gemma solo aporta texto clínico.
- **Métricas clínicas en anotados**: usar exclusivamente las implementaciones de `utils/pkg/chimera_metrics`, sin métricas locales equivalentes.
- **Métricas principales**: Harrell C-index para el ranking de tiempo, media de `time_score` censoring-aware y media de `event_score`.
- **Métricas secundarias**: distribución de errores, cobertura por estado `event`, casos sin pares admisibles, latencia por caso y tasa de errores del agente.
- **Artefactos**: outputs por caso, manifest de casos, logs, `version.txt`, resultados tabulares/JSON y reporte de casos fallidos.
- **Advertencia estadística**: no se aplicará prueba de significancia ni intervalo como evidencia confirmatoria, porque el modelo y la cohorte de evaluación coinciden.

## 7. Expected Results & Decision Rules

- **Flujo aceptado** si todos los casos completan el contrato, los JSON son parseables y los outputs Cox son consistentes con la tool.
- **Flujo parcialmente aceptado** si hay fallos aislados pero son reproducibles, diagnosticados y no afectan la determinación de los valores oficiales en los casos exitosos.
- **Flujo rechazado** si Gemma puede modificar los valores Cox, falta sistemáticamente alguno de los dos archivos, se producen predicciones inválidas o la ejecución no es reproducible.
- Las métricas obtenidas en anotados se reportarán como **sanity check optimista**, no como rendimiento final del sistema.
- Un rendimiento final requerirá una cohorte externa/no vista o los casos oficiales de evaluación del challenge.

## 8. Risks & Mitigations

- ⚠️ **Fuga por evaluación in-sample**: el modelo productivo fue entrenado con los 75 casos. Mitigación: etiquetar todas las métricas como exploratorias y no usarlas para reclamar generalización.
- ⚠️ **Sesgo por ejecución local**: GPU, versión de Ollama o disponibilidad del modelo pueden afectar latencia. Mitigación: registrar imagen Docker, modelo, hardware y versión.
- **Fallo silencioso de casos**: un batch puede ocultar errores individuales. Mitigación: manifest esperado-versus-producido y reporte explícito de fallos.
- **Alteración de valores por el LLM**: el razonamiento podría repetir valores distintos. Mitigación: validar outcome exclusivamente contra la respuesta Cox y mantener reasoning separado.
- **Métrica no comparable**: mezclar implementaciones locales con las oficiales. Mitigación: importar únicamente `chimera_metrics`.

## 9. Reproducibility Checklist

- [ ] Lista exacta de casos descubiertos y anotados registrada
- [ ] GPU, versión de Docker, imagen y `gemma4:e2b` registrados
- [ ] Semilla/configuración relevante registrada, si aplica
- [ ] Hash del código y estado de Git registrados en cada condición
- [ ] `version.txt` creado en cada carpeta de resultados
- [ ] Logs completos de ejecución preservados
- [ ] Outputs originales por caso preservados
- [ ] Versión de `chimera_metrics` registrada
- [ ] Casos fallidos y reintentos documentados
- [ ] Limitación in-sample incluida en el reporte final

## 10. Next Steps

1. Revisar y aceptar este diseño experimental.
2. Elaborar el plan de implementación de `exp_14`, especificando scripts, configuración, ejecución Docker/GPU y cálculo reproducible con `chimera_metrics`.
3. Tras aprobar ese plan, guardarlo como `experiments/exp_14/IMPLEMENTATION.md` antes de modificar archivos o ejecutar la campaña.
4. Ejecutar el experimento y generar posteriormente `experiments/exp_14/reports/summary.md`.
