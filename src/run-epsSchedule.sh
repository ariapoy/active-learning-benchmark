# exp settings
SEED="0";
N_EXP="150";
N_JOBS="1";
# For multi-processing
END=$(($SEED + $N_EXP - 1));

SKD="Linear";
EPS0arr=(0.2 0 0.4 0 0.6 0 0.8 0 1 0 0.05 0.15 0.1 0.3);
EPSTarr=(0 0.2 0 0.4 0 0.6 0 0.8 0 1 0.15 0.05 0.3 0.1);

# small datasets
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
  for ((i=0; i<${#EPS0arr[@]}; i+=1)); do
    echo "$data , ${EPS0arr[i]}, ${SKD}_${EPSTarr[i]}"
    for s in $(seq $SEED $N_JOBS $END); do #
      timeout 25920 python main.py --qs_name eps_greedy --data_set $data --tool libact --hs_name google-zhan --gs_name zhan --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --eps "${EPS0arr[i]}" --schedule "${SKD}_${EPSTarr[i]}"
    done
  done
done

# large datasets
SEED="0";
N_EXP="15";
# For multi-processing
END=$(($SEED + $N_EXP - 1));
for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
  for ((i=0; i<${#EPS0arr[@]}; i+=1)); do
    echo "$data , ${EPS0arr[i]}, ${SKD}_${EPSTarr[i]}"
    for s in $(seq $SEED $N_JOBS $END); do #
      timeout 259200 python main.py --qs_name eps_greedy --data_set $data --tool libact --hs_name google-zhan --gs_name zhan --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --eps "${EPS0arr[i]}" --schedule "${SKD}_${EPSTarr[i]}"
    done
  done
done
