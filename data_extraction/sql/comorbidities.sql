select ad.subject_id, ad.hadm_id, i.stay_id as icustay_id,
    congestive_heart_failure, cardiac_arrhythmias, valvular_disease,
    pulmonary_circulation, peripheral_vascular, hypertension, paralysis,
    other_neurological, chronic_pulmonary, diabetes_uncomplicated,
    diabetes_complicated, hypothyroidism, renal_failure, liver_disease,
    peptic_ulcer, aids, lymphoma, metastatic_cancer, solid_tumor,
    rheumatoid_arthritis, coagulopathy, obesity, weight_loss,
    fluid_electrolyte, blood_loss_anemia, deficiency_anemias, alcohol_abuse,
    drug_abuse, psychoses, depression
from `physionet-data.mimic_core.admissions` as ad, `physionet-data.mimic_icu.icustays` as i, `{}` as elix
where ad.hadm_id=i.hadm_id and elix.hadm_id=ad.hadm_id
order by subject_id asc, intime asc