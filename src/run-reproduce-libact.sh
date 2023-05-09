# exp settings
TASK="zhan";  # task-oriented model
SEED="0";  # starting seed
N_EXP="150";  # number of experiments
# For multi-processing
N_JOBS="1";  # number of process, depending on your resources and algorithms
END=$(($SEED + $N_EXP - 1));  # ending seed

# libact
TOOL="libact";

echo "Small datasets!"
# us-zhan
QS="us-zhan";
QUERY="us-zhan";
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
    for s in $(seq $SEED $N_JOBS $END); do #
      timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done

# qbc
QS="qbc-zhan";
QUERY="qbc-zhan";
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED $N_JOBS $END); do #
      timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done

# albl
QS="albl-zhan";
QUERY="albl-zhan";
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED $N_JOBS $END); do #
      timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done

# kcenter
# dwus
# quire
# hintsvm
QUERY="libact-zhan";
for QS in "dwus-zhan" "hintsvm-zhan" "kcenter-zhan" "quire-zhan"; do
  for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
    echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
    for s in $(seq $SEED $N_JOBS $END); do #
      timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    done
  done
done

# vr
QS="vr-zhan";
QUERY="vr-zhan";
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED $N_JOBS $END); do #
    timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done

echo "Large datasets!"
# us-zhan
QS="us-zhan";
QUERY="us-zhan";
for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
    for s in $(seq $SEED $N_JOBS $END); do #
      timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done

# qbc
QS="qbc-zhan";
QUERY="qbc-zhan";
for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED $N_JOBS $END); do #
      timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done

# albl
QS="albl-zhan";
QUERY="albl-zhan";
for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED $N_JOBS $END); do #
      timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done

# kcenter
# dwus
# quire
# hintsvm
QUERY="libact-zhan";
for QS in "dwus-zhan" "hintsvm-zhan" "kcenter-zhan" "quire-zhan"; do
  for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
    echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
    if [ "$data" == "spambase" ] && [ "$QS" == "quire-zhan" ]; then
        echo "quire-zhan occurs errors on spambase";
    else
      for s in $(seq $SEED $N_JOBS $END); do #
        timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
      done
    fi
  done
done

# vr
QS="vr-zhan";
QUERY="vr-zhan";
for data in "banana"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED $N_JOBS $END); do #
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done
