# exp settings
SEED="0";
N_EXP="150";
# For multi-processing
N_JOBS="5";
END=$(($SEED + $N_EXP - 1));
TOOL="google";
QS="margin-zhan";

# QUERY model
QUERY="LR";
# Task model

# margin-zhan
# for TASK in "RBFSVM" "RandomForest" "LR"; do
for TASK in "RBFSVM" "RandomForest"; do
  # for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
  # for data in "ex8a"; do
  for data in "sonar" "haberman" "ionosphere" "tic" "gcloudb" "appendicitis" "checkerboard" "breast"; do
      echo "Start $QS with $HS of $data on $EXP with $N_JOBS"
      for s in $(seq $SEED $N_JOBS $END); do #
          timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
      done
  done
done
