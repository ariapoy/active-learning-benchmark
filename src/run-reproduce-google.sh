# exp settings
TASK="zhan";  # task-oriented model
SEED="0";  # starting seed
N_EXP="150";  # number of experiments
# For multi-processing
N_JOBS="1";  # number of process, depending on your resources and algorithms
END=$(($SEED + $N_EXP - 1));  # ending seed

# google
TOOL="google";

echo "Small datasets!"
# uniform-zhan
# margin-zhan
# hier-zhan
# mcm-zhan
# graph-zhan
# infodiv-zhan
QUERY="google-zhan";  # query-oriented model
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
  for QS in "uniform-zhan" "margin-zhan" "hier-zhan" "mcm-zhan" "graph-zhan" "infodiv-zhan"; do
    echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
      for s in $(seq $SEED $N_JOBS $END); do #
        timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
      done
  done
done

echo "Large datasets!"
for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
  for QS in "uniform-zhan" "margin-zhan" "hier-zhan" "mcm-zhan" "graph-zhan" "infodiv-zhan"; do
    echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
      for s in $(seq $SEED $N_JOBS $END); do #
        timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
      done
  done
done
