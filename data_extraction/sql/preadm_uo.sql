select distinct oe.stay_id as icustay_id, UNIX_SECONDS(TIMESTAMP(oe.charttime)) as charttime, oe.itemid, oe.value , DATETIME_DIFF(ic.intime, oe.charttime, minute) as datediff_minutes
from `physionet-data.mimic_icu.outputevents` oe, `physionet-data.mimic_icu.icustays` ic
where oe.stay_id=ic.stay_id and itemid in (	40060,	226633)	
order by icustay_id, charttime, itemid