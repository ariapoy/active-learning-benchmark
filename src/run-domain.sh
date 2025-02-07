# src/run-multiclass.sh

TASK="XGBoost" # "XGBoost";  # task-oriented model
QUERY="XGBoost" # "XGBoost";  # query-oriented model
# exp settings
SEED="0";
N_EXP="10";
END=$(($SEED + $N_EXP - 1));
# For multi-processing
N_JOBS="5";

# list datasets and init_lbl_size as array
DATASETS=("cifar10" "imdb")
INIT_LBL_SIZES=(100 20)

# scikital
TOOL="scikital";
for QS in "skal_uniform" "skal_us_margin" "skal_bald" "skal_coreset"; do
  for (( i=0; i<${#DATASETS[*]}; ++i)); do
    for s in $(seq $SEED $N_JOBS $END); do #
      echo "Exps $s for $QS on ${DATASETS[$i]} with $QUERY x $TASK";
      timeout 86400 python main.py --data_set "${DATASETS[$i]}" --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --init_lbl_size "${INIT_LBL_SIZES[$i]}" --n_jobs $N_JOBS --n_trials $N_JOBS --tool $TOOL --scale --total_budget "3000" --init_lbl_type "RS" --batch_size "1" --hyperparams_type "default";
    done
  done
done

# google
TOOL="google";
QS="mcm";
for (( i=0; i<${#DATASETS[*]}; ++i)); do
  for s in $(seq $SEED $N_JOBS $END); do #
    echo "Exps $s for $QS on ${DATASETS[$i]} with $QUERY x $TASK";
    timeout 86400 python main.py --data_set "${DATASETS[$i]}" --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --init_lbl_size "${INIT_LBL_SIZES[$i]}" --n_jobs $N_JOBS --n_trials $N_JOBS --tool $TOOL --scale --total_budget "3000" --init_lbl_type "RS" --batch_size "1" --hyperparams_type "default";
  done
done