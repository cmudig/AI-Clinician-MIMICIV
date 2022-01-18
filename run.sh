#!/bin/bash

command=$1

if [ "$command" = "extract" ]; then

  echo "EXTRACT DATA"
  echo "Using client secret: $2"
  echo "Using BigQuery project: $3"
  echo

  mkdir -p data
  python data_extraction/extract.py $2 $3

  echo "DONE"

elif [ "$command" = "preprocess" ]; then

  echo "PREPROCESS DATA"
  echo "Using final data output directory name (in data/): $2"
  echo

  echo "1/11 PREPROCESS RAW DATA"
  python preprocessing/01_preprocess_raw_data.py || exit 1

  echo "2/11 CALCULATE SEPSIS ONSET"
  python preprocessing/02_calculate_sepsis_onset.py || exit 1

  echo "3/11 BUILD PATIENT STATES - SEPSIS COHORT"
  python preprocessing/03_build_patient_states.py data/intermediates/sepsis_cohort --window-before 49 --window-after 25 || exit 1

  echo "4/11 IMPUTE STATES - SEPSIS COHORT"
  python preprocessing/04_impute_states.py data/intermediates/sepsis_cohort/patient_states.csv data/intermediates/sepsis_cohort/patient_states_filled.csv --mask-file data/intermediates/sepsis_cohort/state_imputation_mask.csv || exit 1

  echo "5/11 BUILD STATES AND ACTIONS - SEPSIS COHORT"
  python preprocessing/05_build_states_and_actions.py data/intermediates/sepsis_cohort/patient_states_filled.csv data/intermediates/sepsis_cohort/qstime.csv data/intermediates/sepsis_cohort/states_and_actions.csv --window-before 49 --window-after 25 --mapping-file data/intermediates/sepsis_cohort/bin_mapping.csv || exit 1

  echo "6/11 IMPUTE STATES AND ACTIONS - SEPSIS COHORT"
  python preprocessing/06_impute_states_actions.py data/intermediates/sepsis_cohort/states_and_actions.csv data/intermediates/sepsis_cohort/states_and_actions_filled.csv --mask-file data/intermediates/sepsis_cohort/states_and_actions_mask.csv || exit 1

  echo "7/11 BUILD SEPSIS COHORT"
  python preprocessing/07_build_sepsis_cohort.py data/intermediates/sepsis_cohort/states_and_actions_filled.csv data/intermediates/sepsis_cohort/qstime.csv data/$2 || exit 1

  echo "8/11 BUILD PATIENT STATES - MDP"
  python preprocessing/03_build_patient_states.py data/intermediates/mdp --window-before 25 --window-after 49 || exit 1

  echo "9/11 IMPUTE STATES - MDP"
  python preprocessing/04_impute_states.py data/intermediates/mdp/patient_states.csv data/intermediates/mdp/patient_states_filled.csv --mask-file data/intermediates/mdp/state_imputation_mask.csv || exit 1

  echo "10/11 BUILD STATES AND ACTIONS - MDP"
  python preprocessing/05_build_states_and_actions.py data/intermediates/mdp/patient_states_filled.csv data/intermediates/mdp/qstime.csv data/intermediates/mdp/states_and_actions.csv --window-before 49 --window-after 25 --mapping-file data/intermediates/mdp/bin_mapping.csv || exit 1

  echo "11/11 IMPUTE STATES AND ACTIONS - MDP"
  python preprocessing/06_impute_states_actions.py data/intermediates/mdp/states_and_actions.csv data/$2/mimic_dataset.csv --mask-file data/intermediates/mdp/states_and_actions_mask.csv || exit 1

  echo "DONE"

elif [ "$command" = "model" ]; then

  echo "BUILD MODELS"
  echo "Using MIMIC dataset directory (in data/): $2"
  echo "Using model output directory (in data/): $3"
  echo

  echo "GENERATE DATASETS"
  python modeling/01_generate_datasets.py data/$2 data/$3 || exit 1

  echo "TRAIN MODELS"
  python modeling/02_train_models.py data/$3 --n-models 100 || exit 1

  echo "DONE"
fi