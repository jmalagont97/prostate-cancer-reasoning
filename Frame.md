
Variables 

It is related with the set of input variables to which the model must be to respond

Tabular model:

cli_age
cli_psa
cli_psap
cli_psav 
cli_psad 
cli_vol 
cli_months 
cli_pirads 
cli_dre 
cli_bx 
cli_cspca 
cli_comorbidity_count
cli_allergies_count
cli_ipss_score
cli_fh_binary
vit_weight_kg 
vit_height_cm 
vit_bmi 
vit_bp_systolic 
vit_bp_diastolic 
vit_heart_rate_bpm 
vit_smoking_status 
vit_smoking_pack_years
path_hist_bx_isup
path_hist_bx_gl_prim
path_hist_bx_gl_sec
psa_tr_count 
psa_tr_first_val 
psa_tr_last_val 
psa_tr_min 
psa_tr_max 
psa_tr_mean 
psa_tr_delta 
psa_tr_slope
lab_creatinine_mg_dl
lab_free_psa_ng_ml 
lab_free_total_ratio

For the pre-processing were normalised by a minmax scaler (continuous variables), categorical (one-hot), without imputation instead of that add a column with a binary value for missing values


- Model for Decision

kNN fuzzy with a n=5, distance=euclidian, spatial_weights=uniform, uncertainty_weights=clear (1), boder_line(0.5), uncertain(0.25) 


- Model for Uncertainty

Obtained results yet unstable and in experimentation




- Model for Variable weights





- Model for reveal sequence

