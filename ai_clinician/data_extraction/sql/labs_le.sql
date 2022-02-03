select xx.icustay_id, UNIX_SECONDS(TIMESTAMP(f.charttime)) as timestp, f.itemid, f.valuenum
from(
select subject_id, hadm_id, stay_id as icustay_id, intime, outtime
from `physionet-data.mimic_icu.icustays`
group by subject_id, hadm_id, icustay_id, intime, outtime
) as xx inner join  `physionet-data.mimic_hosp.labevents` as f on f.hadm_id=xx.hadm_id and DATETIME_DIFF(f.charttime, xx.intime, second) >= 24*3600 and DATETIME_DIFF(xx.outtime, f.charttime, second) >= 24*3600  and f.itemid in  (50971,50822,50824,50806,50931,51081,50885,51003,51222,50810,51301,50983,50902,50809,51006,50912,50960,50893,50808,50804,50878,50861,51464,50883,50976,50862,51002,50889,50811,51221,51279,51300,51265,51275,51274,51237,50820,50821,50818,50802,50813,50882,50803,52167,52166,52165,52923,51624,52647) and valuenum is not null
order by f.hadm_id, timestp, f.itemid