# exp settings
TASK="XGBoost";  # task-oriented model
QUERY="XGBoost";  # query-oriented model
SEED="0";  # starting seed
N_EXP="110";  # number of experiments
# For multi-processing
N_JOBS="5";  # number of process, depending on your resources and algorithms
END=$(($SEED + $N_EXP - 1));  # ending seed

TOOL="alipy";

echo "Small datasets!"
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
  for QS in "eer-zhan" "bmdr-zhan" "spal-zhan"; do
    echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
      for s in $(seq $SEED $N_JOBS $END); do #
        timeout 25920 python main.py --data_set $data --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --tool $TOOL;
      done
  done
done

N_EXP="10";  # number of experiments
BUDGET="3000";
echo "Large datasets!"
for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
  for QS in "eer-zhan" "bmdr-zhan" "spal-zhan"; do
    echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
      for s in $(seq $SEED $N_JOBS $END); do #
        timeout 259200 python main.py --data_set $data --qs_name $QS --total_budget $BUDGET --hs_name $QUERY --gs_name $TASK --total_budget 3000 --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --tool $TOOL;
      done
  done
done
