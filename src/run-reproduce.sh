# exp settings
TASK="zhan";
SEED="0";
N_EXP="150";
# For multi-processing
N_JOBS="1";
END=$(($SEED + $N_EXP - 1));

# 1. BSO
TOOL="bso";  # "google" "libact" "alipy" "bso"
QS="bso-zhan";
BSOCONF="lookDtst";  # for BSO only
QUERY="bso-zhan";
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
    echo "Start $QS with $HS of $data on $EXP with $N_JOBS"
    for s in $(seq $SEED $N_JOBS $END); do #
        timeout 25920 python main.py --tool $TOOL --qs_name $QS --lookDtst $BSOCONF --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    done
done

# # 3. google
TOOL="google";

## 3.1 uniform-zhan
## 3.2 hier-zhan
## 3.3 margin-zhan
## 3.4 graph-zhan
## 3.5 infodiv-zhan
## 3.6 mcm-zhan
QUERY="google-zhan";
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
    for QS in "uniform-zhan" "hier-zhan" "margin-zhan" "graph-zhan" "infodiv-zhan" "mcm-zhan"; do
        echo "Start $QS with $HS of $data on $EXP with $N_JOBS"
        for s in $(seq $SEED $N_JOBS $END); do #
            timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
        done
    done
done

# 2. libact
TOOL="libact";

## 2.1 us-zhan
QS="us-zhan";
QUERY="us-zhan";
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
    echo "Start $QS with $HS of $data on $EXP with $N_JOBS"
    for s in $(seq $SEED $N_JOBS $END); do #
        timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    done
done

## 2.2 qbc
QS="qbc-zhan";
QUERY="qbc-zhan";
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
    echo "Start $QS with $HS of $data on $EXP with $N_JOBS"
    for s in $(seq $SEED $N_JOBS $END); do #
        timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    done
done

## 2.3 albl
QS="albl-zhan";
QUERY="albl-zhan";
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
    echo "Start $QS with $HS of $data on $EXP with $N_JOBS"
    for s in $(seq $SEED $N_JOBS $END); do #
        timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    done
done

## 2.4 dwus
## 2.5 quire
## 2.6 hintsvm
## 2.7 kcenter
QUERY="libact-zhan";
for QS in "dwus-zhan" "hintsvm-zhan" "kcenter-zhan" "quire-zhan"; do
    for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
        echo "Start $QS with $HS of $data on $EXP with $N_JOBS"
        for s in $(seq $SEED $N_JOBS $END); do #
            timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
        done
    done
done

## 2.8 vr
QS="vr-zhan";
QUERY="vr-zhan";
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
    echo "Start $QS with $HS of $data on $EXP with $N_JOBS"
    for s in $(seq $SEED $N_JOBS $END); do #
        timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    done
done

# 4. alipy
TOOL="alipy";

## 4.1 eer-zhan
QS="eer-zhan";
QUERY="eer-zhan";
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
    echo "Start $QS with $HS of $data on $EXP with $N_JOBS"
    for s in $(seq $SEED $N_JOBS $END); do #
        timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    done
done

# 4.4 lal-zhan
QUERY="alipy-zhan"
QS="lal-zhan"
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
    echo "Start $QS with $HS of $data on $EXP with $N_JOBS"
    for s in $(seq $SEED $N_JOBS $END); do #
        timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    done
done

## 4.2 bmdr-zhan
QS="bmdr-zhan"
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
    echo "Start $QS with $HS of $data on $EXP with $N_JOBS"
    for s in $(seq $SEED $N_JOBS $END); do #
        timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    done
done

# 4.3 spal-zhan
QS="spal-zhan"
for data in "appendicitis" "sonar" "parkinsons" "ex8b" "heart" "haberman" "ionosphere" "clean1" "breast" "wdbc" "australian" "diabetes" "mammographic" "ex8a" "tic" "german" "splice" "gcloudb" "gcloudub" "checkerboard"; do
    echo "Start $QS with $HS of $data on $EXP with $N_JOBS"
    for s in $(seq $SEED $N_JOBS $END); do #
        timeout 25920 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    done
done
