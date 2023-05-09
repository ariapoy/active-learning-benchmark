# exp settings
TASK="zhan";  # task-oriented model
SEED="0";  # starting seed
N_EXP="150";  # number of experiments
# For multi-processing
N_JOBS="1";  # number of process, depending on your resources and algorithms
END=$(($SEED + $N_EXP - 1));  # ending seed

# libact
echo "libact infeasible"
TOOL="libact";

echo "Infeasible datasets!"
# vr
QS="vr-zhan";
QUERY="vr-zhan";
for data in "clean1"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED $N_JOBS $END); do #
    timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done

echo "Large datasets!"
# quire
QUERY="libact-zhan";
for QS in "quire-zhan"; do
  for data in "spambase"; do
    echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
    if [ "$data" == "spambase" ] && [ "$qs" == "uniform" ]; then
        echo "quire-zhan occurs errors on spambase";
    fi
    for s in $(seq $SEED $N_JOBS $END); do #
      timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    done
  done
done

# vr
QS="vr-zhan";
QUERY="vr-zhan";
for data in "spambase" "phoneme" "ringnorm" "twonorm" "phishing"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED $N_JOBS $END); do #
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done

# alipy
echo "alipy infeasible"
TOOL="alipy";

echo "Small datasets!"
# spal-zhan
QS="spal-zhan"
for data in "checkerboard"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED $N_JOBS $END); do #
    timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done

echo "Large datasets!"
# eer-zhan
QS="eer-zhan";
QUERY="eer-zhan";
for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
  for QS in "uniform-zhan" "margin-zhan" "hier-zhan" "mcm-zhan" "graph-zhan" "infodiv-zhan"; do
    echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
      for s in $(seq $SEED $N_JOBS $END); do #
        timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
      done
  done
done

# bmdr-zhan
QS="bmdr-zhan"
for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
  for QS in "uniform-zhan" "margin-zhan" "hier-zhan" "mcm-zhan" "graph-zhan" "infodiv-zhan"; do
    echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
      for s in $(seq $SEED $N_JOBS $END); do #
        timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
      done
  done
done

# spal-zhan
QS="spal-zhan"
for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
  for QS in "uniform-zhan" "margin-zhan" "hier-zhan" "mcm-zhan" "graph-zhan" "infodiv-zhan"; do
    echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
      for s in $(seq $SEED $N_JOBS $END); do #
        timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
      done
  done
done

# BSO
echo "bso infeasible"
TOOL="bso";  # "google" "libact" "alipy" "bso"
QS="bso-zhan";
BSOCONF="lookDtst";  # for BSO only
QUERY="bso-zhan";
for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
  echo "Start $QS with $QUERY on $data, repeated $N_EXP times with $N_JOBS process."
  for s in $(seq $SEED $N_JOBS $END); do #
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --lookDtst $BSOCONF --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
  done
done
