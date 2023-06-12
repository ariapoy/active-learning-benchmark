# exp settings
SEED="0";
N_EXP="15";
# For multi-processing
N_JOBS="5";
END=$(($SEED + $N_EXP - 1));
TOOL="google";
QS="margin-zhan";

# QUERY model
QUERY="RBFSVM";
# Task model

## 3.3 margin-zhan
for TASK in "RandomForest" "LR"; do
  for data in "banana" "twonorm"; do
    echo "Start $QS with $HS of $data on $EXP with $N_JOBS"
    for s in $(seq $SEED $N_JOBS $END); do #
      timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    done
  done
done
