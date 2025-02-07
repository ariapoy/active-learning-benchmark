# src/run-multiclass.sh

TASK="LR" # "XGBoost";  # task-oriented model
QUERY="LR" # "XGBoost";  # query-oriented model
# exp settings
SEED="0";
N_EXP="110";
END=$(($SEED + $N_EXP - 1));
# For multi-processing
N_JOBS="5";

# list datasets and init_lbl_size as array
DATASETS=("iris" "vehicle" "wine" "satellite" "winequality" "bean" "academic")
INIT_LBL_SIZES=(3 5 3 6 7 7 3)

# scikital
TOOL="scikital";
for QS in "skal_uniform" "skal_us_margin" "skal_bald" "skal_coreset" "skal_us_ent" "skal_us_lc"; do
  for (( i=0; i<${#DATASETS[*]}; ++i)); do
    echo "One-Shot Exps for $QS on ${DATASETS[$i]} with $QUERY x $TASK";
    for s in $(seq $SEED $N_JOBS $END); do #
      timeout 86400 python main.py --data_set "${DATASETS[$i]}" --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --init_lbl_size "${INIT_LBL_SIZES[$i]}" --n_jobs $N_JOBS --n_trials $N_JOBS --tool $TOOL --scale --total_budget "3000" --init_lbl_type "nShot" --batch_size "1" --hyperparams_type "default";
    done
  done
done

# google
TOOL="google";
QS="mcm"
for (( i=0; i<${#DATASETS[*]}; ++i)); do
  echo "One-Shot Exps for $QS on ${DATASETS[$i]} with $QUERY x $TASK";
  for s in $(seq $SEED $N_JOBS $END); do #
    timeout 86400 python main.py --data_set "${DATASETS[$i]}" --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --init_lbl_size "${INIT_LBL_SIZES[$i]}" --n_jobs $N_JOBS --n_trials $N_JOBS --tool $TOOL --scale --total_budget "3000" --init_lbl_type "nShot" --batch_size "1" --hyperparams_type "default";
  done
done
