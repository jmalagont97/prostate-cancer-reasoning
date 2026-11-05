# CHIMERA: Prostate Cancer Multimodal Reasoning

Framework de investigación y razonamiento agéntico para diagnóstico y toma de decisiones clínicas en patología de cáncer de próstata.

---

## 1. Arquitectura del Repositorio

El repositorio opera bajo un estándar de modularidad estricta y aislamiento por objetivos de investigación (*goals*). La raíz del proyecto contiene únicamente los siguientes elementos canónicos:

```
<project_root>/
├── agent.md       # Configuración de sesión y protocolo de gobierno (untracked)
├── readme.md      # Directrices canónicas del proyecto
├── data/          # Almacén de datos global compartido (orígenes brutos y symlinks promovidos)
└── goals/         # Contenedor modular de líneas de investigación autónomas
```

Cualquier nuevo desarrollo, experimentación o módulo de software debe confinarse dentro de su correspondiente línea de investigación en `goals/<goal_id>/`. Está prohibido agregar código o carpetas dispersas en la raíz del proyecto.

---

## 2. Líneas de Investigación Activas

### Goal 01: `goals/01/` (Pathology Reasoning & Agentic Inference)
Contiene todo el desarrollo, paquetes de software, bancos experimentales y documentación de la Tarea 1 (CHIMERA Pathology Reasoning):

- **`goals/01/pkg/`**: Paquetes internos de software:
  - `pkg/agent/`: Orquestador agéntico (`agent_cli.py`, `schemas.py`, `tools.py`, `ollama_client.py`).
  - `pkg/deploy/`: Motores de inferencia empaquetados (`subtask_1.1`, `subtask_1.2`, `subtask_1.3`).
  - `pkg/embedkit/`: Módulo de procesamiento y extracción de embeddings.
- **`goals/01/experiments/`**: Banco experimental formal:
  - `index.md`: Índice de experimentos y estados (`planned` | `running` | `completed`).
  - `results_summary.md`: Resumen consolidado de resultados empíricos.
  - `exp_1/` a `exp_24/`: 24 experimentos históricos con sus respectivos `DESIGN.md`, `results/` y reportes.
- **`goals/01/docs/`**: Literatura médica, presentaciones y especificaciones de evaluación (`evaluation.md`, `gmu.pdf`, `ici.png`, etc.).
- **`goals/01/utils/labs/`**: Sandbox exclusivo de usuario para pruebas exploratorias rápidas (intocable por agentes de IA).
- **`goals/01/.discussion.md`**: Registro formal de consensos teóricos y metodológicos.
- **`goals/01/.logbook.md`**: Bitácora técnica cronológica de ejecuciones consolidadas.

---

## 3. Directrices Obligatorias para Colaboradores

1. **Agnosticismo de Rutas (Cero Rutas Absolutas):**
   - Prohibido el uso de rutas absolutas (`/home/...`, `/Users/...`).
   - Toda resolución de datos debe ser relativa:
     - Datos compartidos globales: `../../data/` (desde el interior de un goal).
     - Datos propios del goal: `./data/`.
     - Paquetes internos del goal: `./pkg/`.
2. **Ejecución de Cómputo (> 60 segundos):**
   - Los procesos de cómputo intensivo deben ejecutarse de forma desacoplada y persistente en sesiones `tmux` nombradas `<goal_id>_<exp_id>`.
   - Telemetría condensada a una línea por época o hito.
3. **Registro y Trazabilidad:**
   - Todo nuevo experimento debe registrarse en `goals/<goal_id>/experiments/index.md`.
   - Las validaciones experimentales consolidadas deben registrarse en `.logbook.md`.
