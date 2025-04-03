# src/run-multiclass.sh

TASK="XGBoost" # "XGBoost";  # task-oriented model
QUERY="XGBoost" # "XGBoost";  # query-oriented model
# exp settings
SEED="0";
N_EXP="110";
END=$(($SEED + $N_EXP - 1));
# For multi-processing
N_JOBS="5";

# list datasets and init_lbl_size as array
# DATASETS=("heart" "mammographic" "phoneme")
DATASETS=("heart")

# scikital
TOOL="scikital";
for QS in "skal_uniform" "skal_us_margin"; do
  for (( i=0; i<${#DATASETS[*]}; ++i)); do
    for TrainTest in "Fix" "noFix"; do
      for InitSet in "Fix" "noFix"; do
        echo "Exps for $QS on ${DATASETS[$i]} with $TrainTest train-test and $InitSet initial set";
        for s in $(seq $SEED $N_JOBS $END); do #
          timeout 86400 python main.py --data_set "${DATASETS[$i]}" --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --tool $TOOL --scale --total_budget "3000" --init_trn_tst_fix_type $TrainTest --init_set_fix_type $InitSet;
        done
      done
    done
  done
done