# exp settings
SEED="0";
N_EXP="1";
N_JOBS="1";
# For multi-processing
END=$(($SEED + $N_EXP - 1));

# small datasets
for EPS in $(seq 0 0.1 1); do
  for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
    echo "benchmark ($data, $EPS)"
    for s in $(seq $SEED $N_JOBS $END); do #
      timeout 25920 python main.py --qs_name eps_greedy --data_set $data --exp_name "RS_Fix_scale_seed0" --tool libact --hs_name google-zhan --gs_name zhan --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --eps $EPS
    done
  done
done

# large datasets
SEED="0";
N_EXP="15";
# For multi-processing
END=$(($SEED + $N_EXP - 1));
for EPS in $(seq 0 0.1 1); do
  for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
      echo "benchmark ($data, $EPS)"
    for s in $(seq $SEED $N_JOBS $END); do #
      timeout 259200 python main.py --qs_name eps_greedy --data_set $data --tool libact --hs_name google-zhan --gs_name zhan --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --eps $EPS
    done
  done
done
