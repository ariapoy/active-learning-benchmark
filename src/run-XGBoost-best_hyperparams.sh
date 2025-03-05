# exp settings
TASK="XGBoost";  # task-oriented model
QUERY="XGBoost";  # query-oriented model
SEED="0";  # starting seed
N_EXP="110";  # number of experiments
END=$(($SEED + $N_EXP - 1));  # ending seed
# For multi-processing
N_JOBS="5";  # number of process, depending on your resources and algorithms

TOOL="google";

echo "Small datasets!"
for QS in "uniform" "google_us_margin"; do
  for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
    echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
      for s in $(seq $SEED $N_JOBS $END); do #
        timeout 25920 python main.py --data_set $data --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --tool $TOOL;
      done
  done
done

# echo "Large datasets!"
# N_EXP="10";  # number of experiments
# END=$(($SEED + $N_EXP - 1));  # ending seed
# BUDGET="3000";
# for QS in "uniform-zhan" "margin-zhan" "hier-zhan" "mcm-zhan" "graph-zhan" "infodiv-zhan"; do
#   for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing" "pol" "bioresponse" "covertype"; do
#     echo "Start $QS with $QUERY on $data limited budget $BUDGET, repeated $N_EXP times with $N_JOBS process."
#       for s in $(seq $SEED $N_JOBS $END); do #
#         timeout 259200 python main.py --data_set $data --qs_name $QS --total_budget $BUDGET --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --tool $TOOL;
#       done
#   done
# # done