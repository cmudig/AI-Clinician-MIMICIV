-- ------------------------------------------------------------------
-- This query extracts Elixhauser-Quan comorbidity index based on the recorded ICD-9 and ICD-10 codes.
-- Adapted from https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts/comorbidity
--
-- Reference for CCI:
-- (1) Charlson ME, Pompei P, Ales KL, MacKenzie CR. (1987) A new method of classifying prognostic 
-- comorbidity in longitudinal studies: development and validation.J Chronic Dis; 40(5):373-83.
--
-- (2) Charlson M, Szatrowski TP, Peterson J, Gold J. (1994) Validation of a combined comorbidity 
-- index. J Clin Epidemiol; 47(11):1245-51.
-- 
-- Reference for ICD-9-CM and ICD-10 Coding Algorithms for Charlson Comorbidities:
-- (3) Quan H, Sundararajan V, Halfon P, et al. Coding algorithms for defining Comorbidities in ICD-9-CM
-- and ICD-10 administrative data. Med Care. 2005 Nov; 43(11): 1130-9.
-- ------------------------------------------------------------------

WITH diag AS
(
    SELECT 
        hadm_id
        , CASE WHEN icd_version = 9 THEN icd_code ELSE NULL END AS icd9_code
        , CASE WHEN icd_version = 10 THEN icd_code ELSE NULL END AS icd10_code
    FROM `physionet-data.mimic_hosp.diagnoses_icd` diag
)
, com AS
(
    SELECT
        ad.hadm_id

        , MAX(CASE
        when icd9_code in ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493') then 1
        when SUBSTR(icd9_code, 1, 4) in ('4254','4255','4257','4258','4259') then 1
        when SUBSTR(icd9_code, 1, 3) in ('428') then 1
        when SUBSTR(icd10_code, 1, 4) in ('I099', 'I110', 'I130', 'I132', 'I255', 'I420', 'I425', 'I426', 'I427', 'I428', 'I429', 'P290') then 1
        when SUBSTR(icd10_code, 1, 3) in ('I43', 'I50') then 1
        else 0 end) as chf       /* Congestive heart failure */

        , MAX(CASE
        when icd9_code in ('42613','42610','42612','99601','99604') then 1
        when SUBSTR(icd9_code, 1, 4) in ('4260','4267','4269','4270','4271','4272','4273','4274','4276','4278','4279','7850','V450','V533') then 1
        when SUBSTR(icd10_code, 1, 4) in ('I441', 'I442', 'I443', 'I4566', 'I459', 'R000', 'R001', 'R008', 'T821', 'Z450', 'Z950') then 1
        when SUBSTR(icd10_code, 1, 3) in ('I47', 'I48', 'I49') then 1
        else 0 end) as arrhy

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('0932','7463','7464','7465','7466','V422','V433') then 1
        when SUBSTR(icd9_code, 1, 3) in ('394','395','396','397','424') then 1
        when SUBSTR(icd10_code, 1, 4) in ('A520', 'I091', 'I098', 'Q230', 'Q231', 'Q232', 'Q233', 'Z952', 'Z953', 'Z954') then 1
        when SUBSTR(icd10_code, 1, 3) in ('I05', 'I06', 'I07', 'I08', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39') then 1
        else 0 end) as valve     /* Valvular disease */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('4150','4151','4170','4178','4179') then 1
        when SUBSTR(icd9_code, 1, 3) in ('416') then 1
        when SUBSTR(icd10_code, 1, 4) in ('I280', 'I288', 'I289') then 1
        when SUBSTR(icd10_code, 1, 3) in ('I26', 'I27') then 1
        else 0 end) as pulmcirc  /* Pulmonary circulation disorder */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('0930','4373','4431','4432','4438','4439','4471','5571','5579','V434') then 1
        when SUBSTR(icd9_code, 1, 3) in ('440','441') then 1
        when SUBSTR(icd10_code, 1, 4) in ('I731', 'I738', 'I739', 'I771', 'I790', 'I792', 'K551', 'K558', 'K559', 'Z958', 'Z959') then 1
        when SUBSTR(icd10_code, 1, 3) in ('I70', 'I71') then 1
        else 0 end) as perivasc  /* Peripheral vascular disorder */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 3) in ('401') then 1
        when SUBSTR(icd10_code, 1, 3) in ('I10') then 1
        else 0 end) as htn       /* Hypertension, uncomplicated */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 3) in ('402','403','404','405') then 1
        when SUBSTR(icd10_code, 1, 3) in ('I11', 'I12', 'I13', 'I15') then 1
        else 0 end) as htncx     /* Hypertension, complicated */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('3341','3440','3441','3442','3443','3444','3445','3446','3449') then 1
        when SUBSTR(icd9_code, 1, 3) in ('342','343') then 1
        when SUBSTR(icd10_code, 1, 4) in ('G041', 'G114', 'G801', 'G802', 'G830', 'G831', 'G832', 'G833', 'G834', 'G839') then 1
        when SUBSTR(icd10_code, 1, 3) in ('G81', 'G82') then 1
        else 0 end) as para      /* Paralysis */

        , MAX(CASE
        when icd9_code in ('33392') then 1
        when SUBSTR(icd9_code, 1, 4) in ('3319','3320','3321','3334','3335','3362','3481','3483','7803','7843') then 1
        when SUBSTR(icd9_code, 1, 3) in ('334','335','340','341','345') then 1
        when SUBSTR(icd10_code, 1, 4) in ('G254', 'G255', 'G312', 'G318', 'G319', 'G931', 'G934', 'R470') then 1
        when SUBSTR(icd10_code, 1, 3) in ('G10', 'G11', 'G12', 'G13', 'G20', 'G21', 'G22', 'G32', 'G35', 'G36', 'G37', 'G40', 'G41', 'R56') then 1
        else 0 end) as neuro     /* Other neurological */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('4168','4169','5064','5081','5088') then 1
        when SUBSTR(icd9_code, 1, 3) in ('490','491','492','493','494','495','496','500','501','502','503','504','505') then 1
        when SUBSTR(icd10_code, 1, 4) in ('I278', 'I279', 'J684', 'J701', 'J703') then 1
        when SUBSTR(icd10_code, 1, 3) in ('J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47', 'J60', 'J61', 'J62', 'J63', 'J64', 'J65', 'J66', 'J67') then 1
        else 0 end) as chrnlung  /* Chronic pulmonary disease */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('2500','2501','2502','2503') then 1
        when SUBSTR(icd10_code, 1, 4) in ('E100', 'E101', 'E109', 'E110', 'E111', 'E119', 'E120', 'E121', 'E129', 'E130', 'E131', 'E139', 'E140', 'E141', 'E149') then 1
        else 0 end) as dm        /* Diabetes w/o chronic complications*/

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('2504','2505','2506','2507','2508','2509') then 1
        when SUBSTR(icd10_code, 1, 4) in ('E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128', 'E132', 'E133', 'E134', 'E135', 'E136', 'E137', 'E138', 'E142', 'E143', 'E144', 'E145', 'E146', 'E147', 'E148') then 1
        else 0 end) as dmcx      /* Diabetes w/ chronic complications */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('2409','2461','2468') then 1
        when SUBSTR(icd9_code, 1, 3) in ('243','244') then 1
        when SUBSTR(icd10_code, 1, 4) in ('E890') then 1
        when SUBSTR(icd10_code, 1, 3) in ('E0', 'E1', 'E2', 'E3') then 1
        else 0 end) as hypothy   /* Hypothyroidism */

        , MAX(CASE
        when icd9_code in ('40301','40311','40391','40402','40403','40412','40413','40492','40493') then 1
        when SUBSTR(icd9_code, 1, 4) in ('5880','V420','V451') then 1
        when SUBSTR(icd9_code, 1, 3) in ('585','586','V56') then 1
        when SUBSTR(icd10_code, 1, 4) in ('I120', 'I131', 'N250', 'Z490', 'Z491', 'Z492', 'Z940', 'Z992') then 1
        when SUBSTR(icd10_code, 1, 3) in ('N18', 'N19') then 1
        else 0 end) as renlfail  /* Renal failure */

        , MAX(CASE
        when icd9_code in ('07022','07023','07032','07033','07044','07054') then 1
        when SUBSTR(icd9_code, 1, 4) in ('0706','0709','4560','4561','4562','5722','5723','5724','5728','5733','5734','5738','5739','V427') then 1
        when SUBSTR(icd9_code, 1, 3) in ('570','571') then 1
        when SUBSTR(icd10_code, 1, 4) in ('I864', 'I982', 'K711', 'K713', 'K714', 'K715', 'K717', 'K760', 'K762', 'K763', 'K764', 'K765', 'K766', 'K767', 'K768', 'K769', 'Z944') then 1
        when SUBSTR(icd10_code, 1, 3) in ('B18', 'I85', 'K70', 'K72', 'K73', 'K74') then 1
        else 0 end) as liver     /* Liver disease */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('5317','5319','5327','5329','5337','5339','5347','5349') then 1
        when SUBSTR(icd10_code, 1, 4) in ('K257', 'K259', 'K267', 'K269', 'K277', 'K279', 'K287', 'K289') then 1
        else 0 end) as ulcer     /* Chronic Peptic ulcer disease (includes bleeding only if obstruction is also present) */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 3) in ('042','043','044') then 1
        when SUBSTR(icd10_code, 1, 3) in ('B20', 'B21', 'B22', 'B24') then 1
        else 0 end) as aids      /* HIV and AIDS */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('2030','2386') then 1
        when SUBSTR(icd9_code, 1, 3) in ('200','201','202') then 1
        when SUBSTR(icd10_code, 1, 4) in ('C900', 'C902') then 1
        when SUBSTR(icd10_code, 1, 3) in ('C81', 'C82', 'C83', 'C84', 'C85', 'C88', 'C96') then 1
        else 0 end) as lymph     /* Lymphoma */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 3) in ('196','197','198','199') then 1
        when SUBSTR(icd10_code, 1, 3) in ('C77', 'C78', 'C79', 'C80') then 1
        else 0 end) as mets      /* Metastatic cancer */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 3) in
        (
            '140','141','142','143','144','145','146','147','148','149','150','151','152'
            ,'153','154','155','156','157','158','159','160','161','162','163','164','165'
            ,'166','167','168','169','170','171','172','174','175','176','177','178','179'
            ,'180','181','182','183','184','185','186','187','188','189','190','191','192'
            ,'193','194','195'
        ) then 1
        when SUBSTR(icd10_code, 1, 3) in (
            'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'
            , 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14'
            , 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
            , 'C22', 'C23', 'C24', 'C25', 'C26', 'C30', 'C31'
            , 'C32', 'C33', 'C34', 'C37', 'C38', 'C39', 'C40'
            , 'C41', 'C43', 'C45', 'C46', 'C47', 'C48', 'C49'
            , 'C50', 'C51', 'C52', 'C53', 'C54', 'C55', 'C56'
            , 'C57', 'C58', 'C60', 'C61', 'C62', 'C63', 'C64'
            , 'C65', 'C66', 'C67', 'C68', 'C69', 'C70', 'C71'
            , 'C72', 'C73', 'C74', 'C75', 'C76', 'C97') then 1
        else 0 end) as tumor     /* Solid tumor without metastasis */

        , MAX(CASE
        when icd9_code in ('72889','72930') then 1
        when SUBSTR(icd9_code, 1, 4) in ('7010','7100','7101','7102','7103','7104','7108','7109','7112','7193','7285') then 1
        when SUBSTR(icd9_code, 1, 3) in ('446','714','720','725') then 1
        when SUBSTR(icd10_code, 1, 4) in ('L940', 'L941', 'L943', 'M120', 'M123', 'M310', 'M311', 'M312', 'M313', 'M461', 'M468', 'M469') then 1
        when SUBSTR(icd10_code, 1, 3) in ('M05', 'M06', 'M08', 'M30', 'M32', 'M33', 'M34', 'M35', 'M45') then 1
        else 0 end) as arth              /* Rheumatoid arthritis/collagen vascular diseases */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('2871','2873','2874','2875') then 1
        when SUBSTR(icd9_code, 1, 3) in ('286') then 1
        when SUBSTR(icd10_code, 1, 4) in ('D691', 'D693', 'D694', 'D695', 'D696') then 1
        when SUBSTR(icd10_code, 1, 3) in ('D65', 'D66', 'D67', 'D68') then 1
        else 0 end) as coag      /* Coagulation deficiency */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('2780') then 1
        when SUBSTR(icd10_code, 1, 3) in ('E66') then 1
        else 0 end) as obese     /* Obesity      */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('7832','7994') then 1
        when SUBSTR(icd9_code, 1, 3) in ('260','261','262','263') then 1
        when SUBSTR(icd10_code, 1, 4) in ('R634', 'R64') then 1
        when SUBSTR(icd10_code, 1, 3) in ('E40', 'E41', 'E42', 'E43', 'E44', 'E45', 'E46') then 1
        else 0 end) as wghtloss  /* Weight loss */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('2536') then 1
        when SUBSTR(icd9_code, 1, 3) in ('276') then 1
        when SUBSTR(icd10_code, 1, 4) in ('E222') then 1
        when SUBSTR(icd10_code, 1, 3) in ('E86', 'E87') then 1
        else 0 end) as lytes     /* Fluid and electrolyte disorders */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('2800') then 1
        when SUBSTR(icd10_code, 1, 4) in ('D500') then 1
        else 0 end) as bldloss   /* Blood loss anemia */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('2801','2808','2809') then 1
        when SUBSTR(icd9_code, 1, 3) in ('281') then 1
        when SUBSTR(icd10_code, 1, 4) in ('D508', 'D509') then 1
        when SUBSTR(icd10_code, 1, 3) in ('D51', 'D52', 'D53') then 1
        else 0 end) as anemdef  /* Deficiency anemias */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('2652','2911','2912','2913','2915','2918','2919','3030','3039','3050','3575','4255','5353','5710','5711','5712','5713','V113') then 1
        when SUBSTR(icd9_code, 1, 3) in ('980') then 1
        when SUBSTR(icd10_code, 1, 4) in ('F10', 'E52', 'G621', 'I426', 'K292', 'K700', 'K703', 'K709', 'Z502', 'Z714', 'Z721') then 1
        when SUBSTR(icd10_code, 1, 3) in ('T51') then 1
        else 0 end) as alcohol /* Alcohol abuse */

        , MAX(CASE
        when icd9_code in ('V6542') then 1
        when SUBSTR(icd9_code, 1, 4) in ('3052','3053','3054','3055','3056','3057','3058','3059') then 1
        when SUBSTR(icd9_code, 1, 3) in ('292','304') then 1
        when SUBSTR(icd10_code, 1, 4) in ('Z715', 'Z722') then 1
        when SUBSTR(icd10_code, 1, 3) in ('F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F18', 'F19') then 1
        else 0 end) as drug /* Drug abuse */

        , MAX(CASE
        when icd9_code in ('29604','29614','29644','29654') then 1
        when SUBSTR(icd9_code, 1, 4) in ('2938') then 1
        when SUBSTR(icd9_code, 1, 3) in ('295','297','298') then 1
        when SUBSTR(icd10_code, 1, 4) in ('F302', 'F312', 'F315') then 1
        when SUBSTR(icd10_code, 1, 3) in ('F20', 'F22', 'F23', 'F24', 'F25', 'F28', 'F29') then 1
        else 0 end) as psych /* Psychoses */

        , MAX(CASE
        when SUBSTR(icd9_code, 1, 4) in ('2962','2963','2965','3004') then 1
        when SUBSTR(icd9_code, 1, 3) in ('309','311') then 1
        when SUBSTR(icd10_code, 1, 4) in ('F204', 'F313', 'F314', 'F315', 'F341', 'F412', 'F432') then 1
        when SUBSTR(icd10_code, 1, 3) in ('F32', 'F33') then 1
        else 0 end) as depress  /* Depression */
        
    FROM `physionet-data.mimic_core.admissions` ad
    LEFT JOIN diag
    ON ad.hadm_id = diag.hadm_id
    GROUP BY ad.hadm_id
)
-- now merge these flags together to define elixhauser
-- most are straightforward.. but hypertension flags are a bit more complicated
select adm.hadm_id
, chf as congestive_heart_failure
, arrhy as cardiac_arrhythmias
, valve as valvular_disease
, pulmcirc as pulmonary_circulation
, perivasc as peripheral_vascular
-- we combine "htn" and "htncx" into "HYPERTENSION"
, case
    when htn = 1 then 1
    when htncx = 1 then 1
  else 0 end as hypertension
, para as paralysis
, neuro as other_neurological
, chrnlung as chronic_pulmonary
-- only the more severe comorbidity (complicated diabetes) is kept
, case
    when dmcx = 1 then 0
    when dm = 1 then 1
  else 0 end as diabetes_uncomplicated
, dmcx as diabetes_complicated
, hypothy as hypothyroidism
, renlfail as renal_failure
, liver as liver_disease
, ulcer as peptic_ulcer
, aids as aids
, lymph as lymphoma
, mets as metastatic_cancer
-- only the more severe comorbidity (metastatic cancer) is kept
, case
    when mets = 1 then 0
    when tumor = 1 then 1
  else 0 end as solid_tumor
, arth as rheumatoid_arthritis
, coag as coagulopathy
, obese as obesity
, wghtloss as weight_loss
, lytes as fluid_electrolyte
, bldloss as blood_loss_anemia
, anemdef as deficiency_anemias
, alcohol as alcohol_abuse
, drug as drug_abuse
, psych as psychoses
, depress as depression

FROM `physionet-data.mimic_core.admissions` adm
left join com eli
  on adm.hadm_id = eli.hadm_id
order by adm.hadm_id;