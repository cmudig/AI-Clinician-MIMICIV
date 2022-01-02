select stay_id as icustay_id, UNIX_SECONDS(TIMESTAMP(charttime)) as charttime, itemid, value
from `physionet-data.mimic_icu.outputevents`
where stay_id is not null and value is not null and itemid in (40055	,43175	,40069,	40094	,40715	,40473	,40085,	40057,	40056	,40405	,40428,	40096,	40651,226559	,226560	,227510	,226561	,227489	,226584,	226563	,226564	,226565	,226557	,226558)
order by icustay_id, charttime, itemid