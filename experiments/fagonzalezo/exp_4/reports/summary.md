# Monte Carlo Validation Design (exp_4) Summary Report

**Date**: 2026-08-04  
**Total Cohort Checked**: 195 patients  
**Excluded (Problematic)**: 5 patients (missing MRI neural representations)  
**Clean Complete-Case Cohort**: 190 patients  

## Modality Completeness Audit Results
- All clinical tabular features are 100% complete.  
- All textual prompts are 100% complete.  
- MRI features: **5 patients** do not have visual embeddings and were excluded from training splits.  

### Excluded Patient IDs List:
- `PT-pseudo_3646e0a2ae13`  
- `PT-pseudo_4d54f04e26ae`  
- `PT-pseudo_4bfd4ec864d8`  
- `PT-pseudo_7dbdcd6f9064`  
- `PT-pseudo_8636aa471ef7`  

## Monte Carlo Partition Parameters
- **Number of splits (B)**: 100  
- **Random seed state**: `42`  
- **Train size per split**: 70 patients  
- **Validation size per split**: 18 patients  
- **Test size per split**: 102 patients (completely frozen with value `-1`)  

## Stratification Stability Analysis
- **Target Biopsy (Yes / No) Overall Ratio**: 1.5882  
- **Mean Training Biopsy Ratio**: 1.5926 $\pm$ 0.0000  
- **Mean Validation Biopsy Ratio**: 1.5714 $\pm$ 0.0000  

## Output Files Checklist
- [x] Partitions design CSV file: `results/mccv_design.csv` (Shape: (190, 101))  
- [x] Diagnostic visualization: `reports/figures/split_distributions.png`  
