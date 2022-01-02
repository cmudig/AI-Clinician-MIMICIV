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
