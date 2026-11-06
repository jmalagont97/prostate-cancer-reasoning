# Experiments Index — Goal 02

| Exp | Title | Status | Hypothesis (1 line) | Verdict | Date |
|-----|-------|--------|---------------------|---------|------|
| [exp_1](exp_1/DESIGN.md) | Transformación Tabular Compartimentada y EDA Task 1 | completed | Mapeo tabular prefijado recupera ≥98% variables clínicas y aisla reglas discriminantes | Success | 2026-11-05 |
| [exp_2](exp_2/DESIGN.md) | Particionamiento Estratificado 50/50, experimental-design.csv y EDA Comparativo | completed | Partición compuesta 50/50 certifica ausencia de sesgo de selección (SMD <0.10, p >0.05) sin data leakage | Success | 2026-11-05 |
| [exp_3](exp_3/DESIGN.md) | Inducción Neurosimbólica Agéntica (Desacople en 2 Fases, 10 Variables GT, Profundidad Libre) | completed | Árbol neurosimbólico inducido en dos fases (selección -> calibración) sobre 10 variables GT alcanza en Test AUC ≥0.85, Macro-F1 ≥0.80 y Acc ≥85% | Success | 2026-11-05 |
| [exp_4](exp_4/DESIGN.md) | Línea Base CART Clásico sin Restricciones (Task 1.1) | completed | CART voraz sin restricciones sobreajusta en Train (Acc=100%) y exhibe en Test brecha de generalización (AUC ≤0.72, Macro-F1 ≤0.72) inferior al árbol neurosimbólico | Success | 2026-11-05 |
| [exp_5](exp_5/DESIGN.md) | Líneas Base Heurísticas Clínicas Deterministas (Task 1.1) | completed | Reglas heurísticas estándar exhiben compromiso rígido insuperable (H0 colapsa especificidad ≤20%, H1/H2 colapsa sensibilidad ≤62%), siendo inferiores al árbol neurosimbólico (Acc 82.6%, Macro-F1 0.818) | Success | 2026-11-05 |
