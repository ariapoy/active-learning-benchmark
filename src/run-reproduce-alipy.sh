# exp settings
TASK="zhan";  # task-oriented model
SEED="0";  # starting seed
N_EXP="150";  # number of experiments
# For multi-processing
N_JOBS="1";  # number of process, depending on your resources and algorithms
END=$(($SEED + $N_EXP - 1));  # ending seed

# alipy
TOOL="alipy";

echo "Small datasets!"
# eer-zhan
QS="eer-zhan";
QUERY="eer-zhan";
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED $N_JOBS $END); do #
    timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done

# lal-zhan
## WANRING! You need large memory and CPUs to build pre-trained random forest.
QUERY="alipy-zhan"
QS="lal-zhan"
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED "1" $END); do #
    timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs "1" --n_trials "1" --data_set $data;
  done
done

# bmdr-zhan
QS="bmdr-zhan"
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED $N_JOBS $END); do #
    timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done

# spal-zhan
QS="spal-zhan"
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED $N_JOBS $END); do #
    timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done

echo "Large datasets!"
# lal-zhan
## WANRING! You need large memory and CPUs to build pre-trained random forest.
QUERY="alipy-zhan"
QS="lal-zhan"
for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED "1" $END); do #
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs "1" --n_trials "1" --data_set $data;
  done
done
