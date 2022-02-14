ANTIBIOTIC_GSN_CODES = (
    '002542','002543','007371','008873','008877','008879','008880','008935',
    '008941','008942','008943','008944','008983','008984','008990','008991',
    '008992','008995','008996','008998','009043','009046','009065','009066',
    '009136','009137','009162','009164','009165','009171','009182','009189',
    '009213','009214','009218','009219','009221','009226','009227','009235',
    '009242','009263','009273','009284','009298','009299','009310','009322',
    '009323','009326','009327','009339','009346','009351','009354','009362',
    '009394','009395','009396','009509','009510','009511','009544','009585',
    '009591','009592','009630','013023','013645','013723','013724','013725',
    '014182','014500','015979','016368','016373','016408','016931','016932',
    '016949','018636','018637','018766','019283','021187','021205','021735',
    '021871','023372','023989','024095','024194','024668','025080','026721',
    '027252','027465','027470','029325','029927','029928','037042','039551',
    '039806','040819','041798','043350','043879','044143','045131','045132',
    '046771','047797','048077','048262','048266','048292','049835','050442',
    '050443','051932','052050','060365','066295','067471')

CHARTEVENT_CODES = (
    226707, 581, 198, 228096, 211, 220179, 220181, 8368, 220210, 220277, 3655, 
    223761, 220074, 492, 491, 8448, 116, 626, 467, 223835, 190, 470, 220339, 
    224686, 224687, 224697, 224695, 224696, 226730, 580, 220045, 225309, 220052, 
    8441, 3337, 646, 223762, 678, 113, 1372, 3420, 471, 506, 224684, 450, 444, 
    535, 543, 224639, 6701, 225312, 225310, 224422, 834, 1366, 160, 223834, 505, 
    684, 448, 226512, 6, 224322, 8555, 618, 228368, 727, 227287, 224700, 224421, 
    445, 227243, 6702, 8440, 3603, 228177, 194, 3083, 224167, 443, 615, 224691, 
    2566, 51, 52, 654, 455, 456, 3050, 681, 2311, 220059, 220061, 220060, 226732
)

COMORBIDITY_FIELDS = [
    'congestive_heart_failure', 'cardiac_arrhythmias', 'valvular_disease',
    'pulmonary_circulation', 'peripheral_vascular', 'hypertension', 'paralysis',
    'other_neurological', 'chronic_pulmonary', 'diabetes_uncomplicated',
    'diabetes_complicated', 'hypothyroidism', 'renal_failure', 'liver_disease',
    'peptic_ulcer', 'aids', 'lymphoma', 'metastatic_cancer', 'solid_tumor',
    'rheumatoid_arthritis', 'coagulopathy', 'obesity', 'weight_loss',
    'fluid_electrolyte', 'blood_loss_anemia', 'deficiency_anemias', 'alcohol_abuse',
    'drug_abuse', 'psychoses', 'depression'
]

CULTURE_CODES = (
    6035,3333,938,941,942,4855,6043,2929,225401,225437,225444,225451,225454,
    225814,225816,225817,225818,225722,225723,225724,225725,225726,225727,
    225728,225729,225730,225731,225732,225733,227726,70006,70011,70012,70013,
    70014,70016,70024,70037,70041,225734,225735,225736,225768,70055,70057,70060,
    70063,70075,70083,226131,80220
)

INPUTEVENT_CODES = (
    225158,225943,226089,225168,225828,225823,220862,220970,220864,225159,
    220995,225170,225825,227533,225161,227531,225171,225827,225941,225823,
    225825,225941,225825,228341,225827,30018,30021,30015,30296,30020,30066,
    30001,30030,30060,30005,30321,3000630061,30009,30179,30190,30143,30160,
    30008,30168,30186,30211,30353,30159,30007,30185,30063,30094,30352,30014,
    30011,30210,46493,45399,46516,40850,30176,30161,30381,30315,42742,30180,
    46087,41491,30004,42698,42244
)

LABS_CE_CODES = (
    223772, 829, 1535, 227442, 227464, 4195, 3726, 3792, 837, 220645, 4194, 
    3725, 3803, 226534, 1536, 4195, 3726, 788, 220602, 1523, 4193, 3724, 226536,
    3747, 225664, 807, 811, 1529, 220621, 226537, 3744, 781, 1162, 225624, 3737,
    791, 1525, 220615, 3750, 821, 1532, 220635, 786, 225625, 1522, 3746, 816,
    225667, 3766, 777, 787, 770, 3801, 769, 3802, 1538, 848, 225690, 803, 1527, 
    225651, 3807, 1539, 849, 772, 1521, 227456, 3727, 227429, 851, 227444, 814, 
    220228, 813, 220545, 3761, 226540, 4197, 3799, 1127, 1542, 220546, 4200, 
    3834, 828, 227457, 3789, 825, 1533, 227466, 3796, 824, 1286, 1671, 1520, 
    768, 220507, 815, 1530, 227467, 780, 1126, 3839, 4753, 779, 490, 3785, 3838, 
    3837, 778, 3784, 3836, 3835, 776, 224828, 3736, 4196, 3740, 74, 225668, 1531, 
    227443, 1817, 228640, 823, 227686, 220587, 227465, 220224, 226063, 226770, 
    227039, 220235, 226062, 227036
)

LABS_LE_CODES = (
     50971,50822,50824,50806,50931,51081,50885,51003,51222,50810,51301,50983,
     50902,50809,51006,50912,50960,50893,50808,50804,50878,50861,51464,50883,
     50976,50862,51002,50889,50811,51221,51279,51300,51265,51275,51274,51237,
     50820,50821,50818,50802,50813,50882,50803,52167,52166,52165,52923,51624,
     52647
)

MECHVENT_MEASUREMENT_CODES = (
    445, 448, 449, 450, 1340, 1486, 1600, 224687, # minute volume
    639, 654, 681, 682, 683, 684,224685,224684,224686, # tidal volume
    218,436,535,444,459,224697,224695,224696,224746,224747, # High/Low/Peak/Mean/Neg insp force ("RespPressure")
    221,1,1211,1655,2000,226873,224738,224419,224750,227187, # Insp pressure
    543, # PlateauPressure
    5865,5866,224707,224709,224705,224706, # APRV pressure
    60,437,505,506,686,220339,224700, # PEEP
    3459, # high pressure relief
    501,502,503,224702, # PCV
    223,667,668,669,670,671,672, # TCPCV
    157,158,1852,3398,3399,3400,3401,3402,3403,3404,8382,227809,227810, # ETT
    224701, # PSVlevel
)

MECHVENT_CODES = (
    640, # extubated
    720, # vent type
    467, # O2 delivery device
) + MECHVENT_MEASUREMENT_CODES

PREADM_FLUID_CODES = (
    30054,30055,30101,30102,30103,30104,30105,30108,226361,226363,226364,
    226365,226367,226368,226369,226370,226371,226372,226375,226376,227070,
    227071,227072
)

UO_CODES = (
    40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405, 
    40428, 40096, 40651, 226559, 226560, 227510, 226561, 227489, 226584,
    226563, 226564, 226565, 226557, 226558
)

VASO_CODES = (
    30128, 30120, 30051, 221749, 221906, 30119, 30047, 30127, 221289,
    222315, 221662, 30043, 30307
)

def abx(mimiciii=False):
    """
    Antibiotics administration. This is gathered from mimic_hosp.prescriptions
    (mimic_icu.icustays is used to get the stay ID), where prescriptions are 
    filtered to a set of Generic Sequence Numbers (GSN) corresponding to 
    antibiotics.
    """
    if mimiciii:
        base = """
            select
                hadm_id, icustay_id,
                UNIX_SECONDS(TIMESTAMP(startdate)) as startdate,
                UNIX_SECONDS(TIMESTAMP(enddate)) as enddate,
                gsn, ndc, dose_val_rx
            from `physionet-data.mimiciii_clinical.prescriptions`
            where gsn in {gsn}
            order by hadm_id, icustay_id
        """
    else:
        base = """
            select 
                p.hadm_id, 
                i.stay_id as icustay_id, 
                UNIX_SECONDS(TIMESTAMP(p.starttime)) as startdate, 
                UNIX_SECONDS(TIMESTAMP(p.stoptime)) as enddate, 
                gsn, ndc, dose_val_rx
            from `physionet-data.mimic_hosp.prescriptions` as p
                left outer join `physionet-data.mimic_icu.icustays` as i 
                on p.hadm_id=i.hadm_id
            where gsn in {gsn}
            order by hadm_id, icustay_id
        """
    return base.format(gsn=repr(ANTIBIOTIC_GSN_CODES))

def ce(min_stay, max_stay, mimiciii=False):
    """
    Chart events - the bulk of information about a patient's stay, including 
    vital signs, ventilator settings, lab values, code status, mental status, 
    etc. (see MIMIC documentation for table mimic_icu.chartevents).
    """
    query = """
    select distinct {stay_id_field} as icustay_id, UNIX_SECONDS(TIMESTAMP(charttime)) as charttime, itemid, 
        case 
        when lower(value) = 'none' then 0 
        when lower(value) = 'ventilator' then 1 
        when lower(value) in ('cannula', 'nasal cannula', 'high flow nasal cannula') then 2 
        when lower(value) = 'face tent' then 3 
        when lower(value) = 'aerosol-cool' then 4 
        when lower(value) = 'trach mask' then 5 
        when lower(value) = 'hi flow neb' then 6 
        when lower(value) = 'non-rebreather' then 7 
        when lower(value) = '' then 8  
        when lower(value) = 'venti mask' then 9 
        when lower(value) = 'medium conc mask' then 10 
        else valuenum end as valuenum 
    from `{table}` 
    where {stay_id_field}>={min_stay} and {stay_id_field}<{max_stay} and value is not null and itemid in {codes}  
    order by icustay_id, charttime
    """
    
    kwargs = {'min_stay': min_stay, 'max_stay': max_stay, 'codes': repr(CHARTEVENT_CODES)}
    if mimiciii:
        kwargs['table'] = 'physionet-data.mimiciii_clinical.chartevents'
        kwargs['stay_id_field'] = 'icustay_id'
    else:
        kwargs['table'] = 'physionet-data.mimic_icu.chartevents'
        kwargs['stay_id_field'] = 'stay_id'
        
    return query.format(**kwargs)

def comorbidities(elixhauser_table, mimiciii=False):
    """
    Table from which the Elixhauser-Quan score is calculated. This table is
    extracted not to provide explicit features for the MIMIC calculation, but
    rather to inform clinicians about patients' specific comorbidities.
    """
    if mimiciii:
        query = """
            select ad.subject_id, ad.hadm_id, i.icustay_id as icustay_id, {fields}
            from `physionet-data.mimiciii_clinical.admissions` as ad, `physionet-data.mimiciii_clinical.icustays` as i, `{elix}` as elix
            where ad.hadm_id=i.hadm_id and elix.hadm_id=ad.hadm_id
            order by subject_id asc, intime asc
        """
    else:
        query = """
            select ad.subject_id, ad.hadm_id, i.stay_id as icustay_id, {fields}
            from `physionet-data.mimic_core.admissions` as ad, `physionet-data.mimic_icu.icustays` as i, `{elix}` as elix
            where ad.hadm_id=i.hadm_id and elix.hadm_id=ad.hadm_id
            order by subject_id asc, intime asc
        """
    return query.format(elix=elixhauser_table, fields=', '.join(COMORBIDITY_FIELDS))

def culture(mimiciii=False):
    """
    According to Komorowski, these "correspond to blood/urine/CSF/sputum 
    cultures etc". This is extracted from mimic_icu.chartevents, where the item 
    ID is within a set of particular measurement types (see my table 
    derived_data.culture_itemids).
    """
    query = """
        select subject_id, hadm_id, {stay_id_field} as icustay_id, UNIX_SECONDS(TIMESTAMP(charttime)) as charttime, itemid
        from `{table}`
        where itemid in {codes}
        order by subject_id, hadm_id, charttime
    """
    kwargs = {'codes': CULTURE_CODES}
    if mimiciii:
        kwargs['stay_id_field'] = 'icustay_id'
        kwargs['table'] = 'physionet-data.mimiciii_clinical.chartevents'
    else:
        kwargs['stay_id_field'] = 'stay_id'
        kwargs['table'] = 'physionet-data.mimic_icu.chartevents'
    return query.format(**kwargs)

def demog(elixhauser_table, mimiciii=False):
    """
    Demographic information, including dates of admission, discharge, and death,
    as well as comorbidities. This is extracted from several tables, including 
    mimic_core.admissions (admission and discharge times), mimic_icu.icustays 
    (ICU type and timing), mimic_core.patients (gender, age, and date of death),
    and the derived Elixhauser-Quan comorbidiites measure (my 
    derived_data.elixhauser_quan table).
    """
    query = """
        select 
            ad.subject_id,
            ad.hadm_id, 
            i.{stay_id_field} as icustay_id,
            UNIX_SECONDS(TIMESTAMP(ad.admittime)) as admittime,
            UNIX_SECONDS(TIMESTAMP(ad.dischtime)) as dischtime, 
            ROW_NUMBER() over (partition by ad.subject_id order by i.intime asc) as adm_order,
            case
                when i.first_careunit='NICU' then 5
                when i.first_careunit='SICU' then 2
                when i.first_careunit='CSRU' then 4
                when i.first_careunit='CCU' then 6
                when i.first_careunit='MICU' then 1
                when i.first_careunit='TSICU' then 3 end as unit,
            UNIX_SECONDS(TIMESTAMP(i.intime)) as intime,
            UNIX_SECONDS(TIMESTAMP(i.outtime)) as outtime, 
            i.los,
            {age} as age,
            {dob} as dob,
            UNIX_SECONDS(TIMESTAMP(p.dod)) as dod,
            p.dod is not NULL as expire_flag,
            case
                when p.gender='M' then 1
                when p.gender='F' then 2 end as gender,
            CAST(DATETIME_DIFF(p.dod, ad.dischtime, second)<=24*3600 and p.dod is not NULL  as int ) as morta_hosp, --died in hosp if recorded DOD is close to hosp discharge
            CAST(DATETIME_DIFF(p.dod, i.intime, second)<=90*24*3600 and p.dod is not NULL  as int ) as morta_90,
            congestive_heart_failure+cardiac_arrhythmias+valvular_disease+pulmonary_circulation+peripheral_vascular+hypertension+paralysis+other_neurological+chronic_pulmonary+diabetes_uncomplicated+diabetes_complicated+hypothyroidism+renal_failure+liver_disease+peptic_ulcer+aids+lymphoma+metastatic_cancer+solid_tumor+rheumatoid_arthritis+coagulopathy+obesity	+weight_loss+fluid_electrolyte+blood_loss_anemia+	deficiency_anemias+alcohol_abuse+drug_abuse+psychoses+depression as elixhauser
        from `{admissions}` as ad, `{icustays}` as i, `{patients}` as p, `{elix}` as elix
        where ad.hadm_id=i.hadm_id and p.subject_id=i.subject_id and elix.hadm_id=ad.hadm_id
        order by subject_id asc, intime asc
    """
    kwargs = {'elix': elixhauser_table}
    if mimiciii:
        kwargs['stay_id_field'] = 'icustay_id'
        kwargs['admissions'] = 'physionet-data.mimiciii_clinical.admissions'
        kwargs['icustays'] = 'physionet-data.mimiciii_clinical.icustays'
        kwargs['patients'] = 'physionet-data.mimiciii_clinical.patients'
        kwargs['dob'] = 'p.dob'
        kwargs['age'] = 'CAST(TIMESTAMP_DIFF(TIMESTAMP(i.intime), TIMESTAMP(p.dob), HOUR) / 8760 AS int)'
    else:
        kwargs['stay_id_field'] = 'stay_id'
        kwargs['admissions'] = 'physionet-data.mimic_core.admissions'
        kwargs['icustays'] = 'physionet-data.mimic_icu.icustays'
        kwargs['patients'] = 'physionet-data.mimic_core.patients'
        kwargs['dob'] = 'p.anchor_year-p.anchor_age'
        kwargs['age'] = 'EXTRACT(year from i.intime) - p.anchor_year + p.anchor_age'        
    return query.format(**kwargs)

def fluid_mv(mimiciii=False):
    """
    Real-time fluid input from Metavision. From the original Komorowski data
    extraction pipeline: 
        * Records with no rate = STAT
        * Records with rate = INFUSION
        * fluids corrected for tonicity
    """
    query = """
        with t1 as
        (
            select
                {stay_id_field} as icustay_id,
                UNIX_SECONDS(TIMESTAMP(starttime)) as starttime,
                UNIX_SECONDS(TIMESTAMP(endtime)) as endtime,
                itemid, amount, rate,
                case
                    when itemid in (30176,30315) then amount *0.25
                    when itemid in (30161) then amount *0.3
                    when itemid in (30020,30015,225823,30321,30186,30211,30353,42742,42244,225159) then amount *0.5 --
                    when itemid in (227531) then amount *2.75
                    when itemid in (30143,225161) then amount *3
                    when itemid in (30009,220862) then amount *5
                    when itemid in (30030,220995,227533) then amount *6.66
                    when itemid in (228341) then amount *8
                    else amount end as tev -- total equivalent volume
            from `{table}`
            -- only real time items !!
            where {stay_id_field} is not null and amount is not null and itemid in {items}
        )
        select
            icustay_id,
            starttime, 
            endtime,
            itemid, 
            round(cast(amount as numeric),3) as amount,
            round(cast(rate as numeric),3) as rate,
            round(cast(tev as numeric),3) as tev -- total equiv volume
        from t1
        order by icustay_id, starttime, itemid
    """
    kwargs = {'items': repr(INPUTEVENT_CODES)}
    if mimiciii:
        kwargs['stay_id_field'] = 'icustay_id'
        kwargs['table'] = 'physionet-data.mimiciii_clinical.inputevents_mv'
    else:
        kwargs['stay_id_field'] = 'stay_id'
        kwargs['table'] = 'physionet-data.mimic_icu.inputevents'
    return query.format(**kwargs)

def fluid_cv(mimiciii=False):
    if not mimiciii:
        return None
    query = """
        with t1 as
        (
            select icustay_id,
                UNIX_SECONDS(TIMESTAMP(charttime)) as charttime,
                itemid,
                amount,
                case when itemid in (30176,30315) then amount *0.25
                when itemid in (30161) then amount *0.3
                when itemid in (30020,30321, 30015,225823,30186,30211,30353,42742,42244,225159,225159,225159) then amount *0.5
                when itemid in (227531) then amount *2.75
                when itemid in (30143,225161) then amount *3
                when itemid in (30009,220862) then amount *5
                when itemid in (30030,220995,227533) then amount *6.66
                when itemid in (228341) then amount *8
                else amount end as tev -- total equivalent volume
            from `physionet-data.mimiciii_clinical.inputevents_cv`
            -- only RT itemids
            where amount is not null and itemid in {items}
            order by icustay_id, charttime, itemid
        )
        select
            icustay_id,
            charttime,
            itemid,
            round(cast(amount as numeric),3) as amount,
            round(cast(tev as numeric),3) as tev -- total equivalent volume
        from t1

    """
    return query.format(items=INPUTEVENT_CODES)
    
def labs_ce(mimiciii=False):
    """Lab events extracted from the chartevents table."""
    query = """
        select
            {stay_id_field} as icustay_id,
            UNIX_SECONDS(TIMESTAMP(charttime)) as charttime,
            itemid,
            valuenum
        from `{table}`
        where valuenum is not null and {stay_id_field} is not null and itemid in {codes}
        order by icustay_id, charttime, itemid
    """
    kwargs = {'codes': repr(LABS_CE_CODES)}
    if mimiciii:
        kwargs['stay_id_field'] = 'icustay_id'
        kwargs['table'] = 'physionet-data.mimiciii_clinical.chartevents'
    else:
        kwargs['stay_id_field'] = 'stay_id'
        kwargs['table'] = 'physionet-data.mimic_icu.chartevents'
    return query.format(**kwargs)

def labs_le(mimiciii=False):
    """Lab events extracted from the labevents table."""
    query = """
        select
            xx.icustay_id,
            UNIX_SECONDS(TIMESTAMP(f.charttime)) as timestp, 
            f.itemid, 
            f.valuenum
        from (
            select subject_id, hadm_id, {stay_id_field} as icustay_id, intime, outtime
            from `{stays}`
            group by subject_id, hadm_id, icustay_id, intime, outtime
        ) as xx inner join `{events}` as f
        on f.hadm_id=xx.hadm_id and
            DATETIME_DIFF(f.charttime, xx.intime, second) >= 24*3600 and
            DATETIME_DIFF(xx.outtime, f.charttime, second) >= 24*3600  and
            f.itemid in {codes} and valuenum is not null
        order by f.hadm_id, timestp, f.itemid
    """
    kwargs = {'codes': repr(LABS_LE_CODES)}
    if mimiciii:
        kwargs['stay_id_field'] = 'icustay_id'
        kwargs['stays'] = 'physionet-data.mimiciii_clinical.icustays'
        kwargs['events'] = 'physionet-data.mimiciii_clinical.labevents'
    else:
        kwargs['stay_id_field'] = 'stay_id'
        kwargs['stays'] = 'physionet-data.mimic_icu.icustays'
        kwargs['events'] = 'physionet-data.mimic_hosp.labevents'
    return query.format(**kwargs)

def mechvent_pe(mimiciii=False):
    """
    Mechanical ventilation information, extracted from the procedureevents
    table (MIMIC-IV only).
    """
    if mimiciii: return None
    return """
        select subject_id,
            hadm_id,
            stay_id,
            UNIX_SECONDS(TIMESTAMP(starttime)) as starttime,
            UNIX_SECONDS(TIMESTAMP(endtime)) as endtime,
            case when itemid in (225792, 225794, 224385, 225433) then 1 else 0 end as mechvent,
            case when itemid in (227194, 227712, 225477, 225468) then 1 else 0 end as extubated,
            case when itemid = 225468 then 1 else 0 end as selfextubated,
            itemid,
            case when valueuom = 'hour' then value * 60
            when valueuom = 'min' then value
            when valueuom = 'day' then value * 60 * 24
            else value end as value
            from `physionet-data.mimic_icu.procedureevents` where itemid in (225792, 225794, 227194, 227712, 224385, 225433, 225468, 225477)
    """

def mechvent(mimiciii=False):
    """
    Default mechanical ventilation information, extracted from chartevents. This
    is supplemented by data from the procedureevents table in MIMIC-IV.
    """
    query = """
        select
            {stay_id_field} as icustay_id, UNIX_SECONDS(TIMESTAMP(charttime)) as charttime    -- case statement determining whether it is an instance of mech vent
            , max(
            case
                when itemid is null or value is null then 0 -- can't have null values
                when itemid = 720 and value != 'Other/Remarks' THEN 1  -- VentTypeRecorded
                when itemid = 467 and value = 'Ventilator' THEN 1 -- O2 delivery device == ventilator
                when itemid in {measurement_codes}
                THEN 1
                else 0
            end
            ) as MechVent
            , max(
                case when itemid is null or value is null then 0
                when itemid = 640 and value = 'Extubated' then 1
                when itemid = 640 and value = 'Self Extubation' then 1
                else 0
                end
                )
                as Extubated
            , max(
                case when itemid is null or value is null then 0
                when itemid = 640 and value = 'Self Extubation' then 1
                else 0
                end
                )
                as SelfExtubated
        from `{events}` ce
        where value is not null
        and itemid in {codes}
        group by icustay_id, charttime
    """
    kwargs = {'codes': repr(MECHVENT_CODES), 'measurement_codes': repr(MECHVENT_MEASUREMENT_CODES)}
    if mimiciii:
        kwargs['stay_id_field'] = 'icustay_id'
        kwargs['events'] = 'physionet-data.mimiciii_clinical.chartevents'
    else:
        kwargs['stay_id_field'] = 'stay_id'
        kwargs['events'] = 'physionet-data.mimic_hosp.chartevents'
    return query.format(**kwargs)

def microbio(mimiciii=False):
    """
    Date and time of all microbiology events (whether they are positive or
    negative). According to the MIMIC documentation:

        Microbiology tests are a common procedure to check for infectious growth 
        and to assess which antibiotic treatments are most effective. If a blood
        culture is requested for a patient, then a blood sample will be taken 
        and sent to the microbiology lab. The time at which this blood sample is 
        taken is the charttime. The spec_type_desc will indicate that this is a 
        blood sample. Bacteria will be cultured on the blood sample, and the 
        remaining columns depend on the outcome of this growth:

        - If no growth is found, the remaining columns will be NULL
        - If bacteria is found, then each organism of bacteria will be present 
        in org_name, resulting in multiple rows for the single specimen (i.e. 
        multiple rows for the given spec_type_desc).
        - If antibiotics are tested on a given bacterial organism, then each 
        antibiotic tested will be present in the ab_name column (i.e. multiple 
        rows for the given org_name associated with the given spec_type_desc). 
        Antibiotic parameters and sensitivities are present in the remaining 
        columns (dilution_text, dilution_comparison, dilution_value, 
        interpretation).
    """
    query = """
        select
            m.subject_id, 
            m.hadm_id, 
            i.{stay_id_field} as icustay_id, 
            UNIX_SECONDS(TIMESTAMP(m.charttime)) as charttime, 
            UNIX_SECONDS(TIMESTAMP(m.chartdate)) as chartdate, 
            spec_itemid, ab_itemid, interpretation
        from `{events}` m left OUTER JOIN `{stays}` i on m.subject_id=i.subject_id and m.hadm_id=i.hadm_id
    """
    kwargs = {}
    if mimiciii:
        kwargs['stay_id_field'] = 'icustay_id'
        kwargs['stays'] = 'physionet-data.mimiciii_clinical.icustays'
        kwargs['events'] = 'physionet-data.mimiciii_clinical.microbiologyevents'
    else:
        kwargs['stay_id_field'] = 'stay_id'
        kwargs['stays'] = 'physionet-data.mimic_icu.icustays'
        kwargs['events'] = 'physionet-data.mimic_hosp.microbiologyevents'
    return query.format(**kwargs)

def preadm_fluid(mimiciii=False):
    """
    Pre-admission fluid intake, as measured from the
    physionet-data.mimic_icu.inputevents table.
    """
    if mimiciii:
        # We need to query both Metavision and CareVue
        query = """
            with mv as (
                select ie.icustay_id, sum(ie.amount) as sum
                from `physionet-data.mimiciii_clinical.inputevents_mv` ie, `physionet-data.mimiciii_clinical.d_items` ci
                where ie.itemid=ci.itemid and ie.itemid in {codes}
                group by icustay_id
            ), cv as (
                select ie.icustay_id, sum(ie.amount) as sum
                from `physionet-data.mimiciii_clinical.inputevents_cv` ie, `physionet-data.mimiciii_clinical.d_items` ci
                where ie.itemid=ci.itemid and ie.itemid in {codes}
                group by icustay_id
            )
            select pt.icustay_id,
                case when mv.sum is not null then mv.sum
                when cv.sum is not null then cv.sum
                else null end as inputpreadm
            from `physionet-data.mimiciii_clinical.icustays` pt
            left outer join mv on mv.icustay_id=pt.icustay_id
            left outer join cv on cv.icustay_id=pt.icustay_id
            order by icustay_id
        """
    else:
        # Metavision only
        query = """
            with mv as
            (
                select ie.stay_id as icustay_id, sum(ie.amount) as sum
                from `physionet-data.mimic_icu.inputevents` ie, `physionet-data.mimic_icu.d_items` ci
                where ie.itemid=ci.itemid and ie.itemid in {codes}
                group by icustay_id
            )
            select pt.stay_id as icustay_id,
                case when mv.sum is not null then mv.sum
                else null end as inputpreadm
            from `physionet-data.mimic_icu.icustays` pt
            left outer join mv
            on mv.icustay_id=pt.stay_id
            order by icustay_id
        """
    kwargs = {'codes': repr(PREADM_FLUID_CODES)}
    return query.format(**kwargs)

def preadm_uo(mimiciii=False):
    """
    Pre-admission output events - information regarding patient outputs
    including urine, drainage, and so on (MIMIC documentation). There is only
    one item ID selected here, which is "Pre-Admission" (all lumped into one
    value).
    """
    query = """
        select distinct oe.{stay_id_field} as icustay_id,
            UNIX_SECONDS(TIMESTAMP(oe.charttime)) as charttime,
            oe.itemid,
            oe.value,
            DATETIME_DIFF(ic.intime, oe.charttime, minute) as datediff_minutes
        from `{events}` oe, `{stays}` ic
        where oe.{stay_id_field}=ic.{stay_id_field} and itemid in (40060, 226633)	
        order by icustay_id, charttime, itemid
    """
    kwargs = {}
    if mimiciii:
        kwargs['stay_id_field'] = 'icustay_id'
        kwargs['stays'] = 'physionet-data.mimiciii_clinical.icustays'
        kwargs['events'] = 'physionet-data.mimiciii_clinical.outputevents'
    else:
        kwargs['stay_id_field'] = 'stay_id'
        kwargs['stays'] = 'physionet-data.mimic_icu.icustays'
        kwargs['events'] = 'physionet-data.mimic_icu.outputevents'
    return query.format(**kwargs)

def uo(mimiciii=False):
    """
    Real-time urine output events from mimic_icu.outputevents.
    """
    query = """
        select {stay_id_field} as icustay_id,\
            UNIX_SECONDS(TIMESTAMP(charttime)) as charttime,
            itemid,
            value
        from `{events}`
        where {stay_id_field} is not null and value is not null and itemid in {codes}
        order by icustay_id, charttime, itemid
    """
    kwargs = {'codes': repr(UO_CODES)}
    if mimiciii:
        kwargs['stay_id_field'] = 'icustay_id'
        kwargs['stays'] = 'physionet-data.mimiciii_clinical.icustays'
        kwargs['events'] = 'physionet-data.mimiciii_clinical.outputevents'
    else:
        kwargs['stay_id_field'] = 'stay_id'
        kwargs['stays'] = 'physionet-data.mimic_icu.icustays'
        kwargs['events'] = 'physionet-data.mimic_icu.outputevents'
    return query.format(**kwargs)

def vaso_base(mv, mimiciii=False):
    """
    Real-time vasopressor input from Metavision. From the original Komorowski
    data extraction code:
    * Drugs converted in noradrenaline-equivalent
    * Body weight assumed 80 kg when missing
    
    Drugs selected are epinephrine, dopamine, phenylephrine, norepinephrine,
    vasopressin. CareVue also contains Levophed and Neosynephrine (extracted in
    MIMIC-III only)."
    """
    query = """
        select {stay_id_field} as icustay_id,
            itemid, 
            {times},
            case when itemid in (30120,221906,30047) and rateuom='mcg/kg/min' then round(cast(rate as numeric),3)  -- norad
                when itemid in (30120,221906,30047) and rateuom='mcg/min' then round(cast(rate/80 as numeric),3)  -- norad
                when itemid in (30119,221289) and rateuom='mcg/kg/min' then round(cast(rate as numeric),3) -- epi
                when itemid in (30119,221289) and rateuom='mcg/min' then round(cast(rate/80 as numeric),3) -- epi
                when itemid in (30051,222315) and rate > 0.2 then round(cast(rate*5/60  as numeric),3) -- vasopressin, in U/h
                when itemid in (30051,222315) and rateuom='units/min' then round(cast(rate*5 as numeric),3) -- vasopressin
                when itemid in (30051,222315) and rateuom='units/hour' then round(cast(rate*5/60 as numeric),3) -- vasopressin
                when itemid in (30128,221749,30127) and rateuom='mcg/kg/min' then round(cast(rate*0.45 as numeric),3) -- phenyl
                when itemid in (30128,221749,30127) and rateuom='mcg/min' then round(cast(rate*0.45 / 80 as numeric),3) -- phenyl
                when itemid in (221662,30043,30307) and rateuom='mcg/kg/min' then round(cast(rate*0.01 as numeric),3)  -- dopa
                when itemid in (221662,30043,30307) and rateuom='mcg/min' then round(cast(rate*0.01/80 as numeric),3) else null end as rate_std-- dopa
        from `{events}`
        where itemid in {codes} and rate is not null {mv_conditions}
        order by icustay_id, itemid, {sort}
    """
    kwargs = {'codes': repr(VASO_CODES)}
    if mv:
        kwargs['mv_conditions'] = "and statusdescription <> 'Rewritten'"
        kwargs['times'] = "UNIX_SECONDS(TIMESTAMP(starttime)) as starttime, UNIX_SECONDS(TIMESTAMP(endtime)) as endtime"
        kwargs['sort'] = "starttime"
    else:
        kwargs['mv_conditions'] = ""
        kwargs['times'] = "UNIX_SECONDS(TIMESTAMP(charttime)) as charttime"
        kwargs['sort'] = "charttime"

    if mimiciii:
        kwargs['stay_id_field'] = 'icustay_id'
        kwargs['stays'] = 'physionet-data.mimiciii_clinical.icustays'
        if mv:
            kwargs['events'] = 'physionet-data.mimiciii_clinical.inputevents_mv'
        else:
            kwargs['events'] = 'physionet-data.mimiciii_clinical.inputevents_cv'
    else:
        kwargs['stay_id_field'] = 'stay_id'
        kwargs['stays'] = 'physionet-data.mimic_icu.icustays'
        kwargs['events'] = 'physionet-data.mimic_icu.inputevents'
    return query.format(**kwargs)

def vaso_mv(mimiciii=False):
    return vaso_base(True, mimiciii)

def vaso_cv(mimiciii=False):
    if not mimiciii: return None
    return vaso_base(False, mimiciii)

def notes(mimiciii=False):
    if not mimiciii: return None
    query = """
        SELECT 
            cn.subject_id as subject_id,
            cn.hadm_id as hadm_id,
            st.icustay_id as icustay_id,
            cn.chartdate as chartdate,
            cn.charttime as charttime,
            cn.category as category,
            cn.text as text
        FROM `physionet-data.mimiciii_notes.noteevents` cn
        LEFT JOIN `physionet-data.mimiciii_clinical.icustays` st
        ON cn.hadm_id = st.hadm_id
        WHERE
            cn.iserror IS NULL AND
            st.icustay_id IS NOT NULL AND 
            (cn.category = "Discharge summary" OR (
                TIMESTAMP_DIFF(cn.chartdate, st.intime, HOUR) >= 0 AND 
                TIMESTAMP_DIFF(cn.chartdate, st.outtime, HOUR) <= 48
            ))
    """    
    return query

SQL_QUERY_FUNCTIONS = {
    "abx": abx,
    "ce": ce,
    "comorbidities": comorbidities,
    "culture": culture,
    "demog": demog,
    "fluid_mv": fluid_mv,
    "fluid_cv": fluid_cv,
    "labs_ce": labs_ce,
    "labs_le": labs_le,
    "mechvent_pe": mechvent_pe,
    "mechvent": mechvent,
    "microbio": microbio,
    "preadm_fluid": preadm_fluid,
    "preadm_uo": preadm_uo,
    "uo": uo,
    "vaso_mv": vaso_mv,
    "vaso_cv": vaso_cv,
    "notes": notes,
}