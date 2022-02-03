select ad.subject_id, ad.hadm_id, i.stay_id as icustay_id ,UNIX_SECONDS(TIMESTAMP(ad.admittime)) as admittime, UNIX_SECONDS(TIMESTAMP(ad.dischtime)) as dischtime, ROW_NUMBER() over (partition by ad.subject_id order by i.intime asc) as adm_order, case when i.first_careunit='NICU' then 5 when i.first_careunit='SICU' then 2 when i.first_careunit='CSRU' then 4 when i.first_careunit='CCU' then 6 when i.first_careunit='MICU' then 1 when i.first_careunit='TSICU' then 3 end as unit,  UNIX_SECONDS(TIMESTAMP(i.intime)) as intime, UNIX_SECONDS(TIMESTAMP(i.outtime)) as outtime, i.los,
EXTRACT(year from i.intime) - p.anchor_year + p.anchor_age as age, p.anchor_year-p.anchor_age as dob, UNIX_SECONDS(TIMESTAMP(p.dod)) as dod,
p.dod is not NULL as expire_flag,  case when p.gender='M' then 1 when p.gender='F' then 2 end as gender,
CAST(DATETIME_DIFF(p.dod, ad.dischtime, second)<=24*3600 and p.dod is not NULL  as int )as morta_hosp,  --died in hosp if recorded DOD is close to hosp discharge
CAST(DATETIME_DIFF(p.dod, i.intime, second)<=90*24*3600 and p.dod is not NULL  as int )as morta_90,
congestive_heart_failure+cardiac_arrhythmias+valvular_disease+pulmonary_circulation+peripheral_vascular+hypertension+paralysis+other_neurological+chronic_pulmonary+diabetes_uncomplicated+diabetes_complicated+hypothyroidism+renal_failure+liver_disease+peptic_ulcer+aids+lymphoma+metastatic_cancer+solid_tumor+rheumatoid_arthritis+coagulopathy+obesity	+weight_loss+fluid_electrolyte+blood_loss_anemia+	deficiency_anemias+alcohol_abuse+drug_abuse+psychoses+depression as elixhauser
from `physionet-data.mimic_core.admissions` as ad, `physionet-data.mimic_icu.icustays` as i, `physionet-data.mimic_core.patients` as p, `{}` as elix
where ad.hadm_id=i.hadm_id and p.subject_id=i.subject_id and elix.hadm_id=ad.hadm_id
order by subject_id asc, intime asc