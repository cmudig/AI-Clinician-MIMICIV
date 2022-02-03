with mv as
(
select ie.stay_id as icustay_id, sum(ie.amount) as sum
from `physionet-data.mimic_icu.inputevents` ie, `physionet-data.mimic_icu.d_items` ci
where ie.itemid=ci.itemid and ie.itemid in (30054,30055,30101,30102,30103,30104,30105,30108,226361,226363,226364,226365,226367,226368,226369,226370,226371,226372,226375,226376,227070,227071,227072)
group by icustay_id
)
select pt.stay_id as icustay_id,
case when mv.sum is not null then mv.sum
else null end as inputpreadm
from `physionet-data.mimic_icu.icustays` pt
left outer join mv
on mv.icustay_id=pt.stay_id
order by icustay_id