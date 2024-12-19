# src/run-multiclass.sh

TASK="XGBoost";  # task-oriented model
QUERY="XGBoost";  # query-oriented model
# exp settings
SEED="0";
N_EXP="20" # "110";
END=$(($SEED + $N_EXP - 1));
# For multi-processing
N_JOBS="10";

# iris
data="iris";
n_init_lbl=30;

# list datasets and init_lbl_size as array
DATASETS=("iris" "vehicle" "wine" "satellite" "winequality" "myocardial" "bean" "academic" "cifar10" "imdb")
INIT_LBL_SIZES=(30 50 30 60 70 80 70 30 100 20)

## scikital
TOOL="scikital";
for QS in "skal_uniform" "skal_us_margin"; do
    for (( i=0; i<${#DATASETS[*]}; ++i)); do
      echo "Exps for $QS on ${DATASETS[$i]} with $QUERY x $TASK";
      for s in $(seq $SEED $N_JOBS $END); do #
        timeout 86400 python main.py --data_set "${DATASETS[$i]}" --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --init_lbl_size "${INIT_LBL_SIZES[$i]}" --n_jobs $N_JOBS --n_trials $N_JOBS --tool $TOOL --scale;
        # echo "python main.py --data_set "${DATASETS[$i]}" --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --init_lbl_size "${INIT_LBL_SIZES[$i]}" --n_jobs $N_JOBS --n_trials $N_JOBS --tool $TOOL --scale;"
      done
    done
done