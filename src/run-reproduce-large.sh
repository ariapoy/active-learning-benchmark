TASK="zhan";
s="0";
N_EXP="15";
N_JOBS="1";
for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
    # google
    TOOL="google";
    QUERY="google-zhan";
    for QS in "uniform-zhan" "hier-zhan" "margin-zhan" "graph-zhan" "infodiv-zhan" "mcm-zhan"; do
        echo "Start $QS with $QUERY on $data"
        timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    done

    # libact
    TOOL="libact";
    QS="us-zhan";
    QUERY="us-zhan";
    echo "Start $QS with $QUERY on $data"
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    QS="qbc-zhan";
    QUERY="qbc-zhan";
    echo "Start $QS with $QUERY on $data"
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    QS="albl-zhan";
    QUERY="albl-zhan";
    echo "Start $QS with $QUERY on $data"
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    QUERY="libact-zhan";
    for QS in "dwus-zhan" "hintsvm-zhan" "kcenter-zhan"; do
        echo "Start $QS with $QUERY on $data"
        timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    done
    QS="vr-zhan";
    QUERY="vr-zhan";
    echo "Start $QS with $QUERY on $data"
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    
    # alipy
    TOOL="alipy";
    QS="eer-zhan";
    QUERY="eer-zhan";
    echo "Start $QS with $QUERY on $data"
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    QS="lal-zhan"
    QUERY="alipy-zhan"
    echo "Start $QS with $QUERY on $data"
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    QS="bmdr-zhan"
    echo "Start $QS with $QUERY on $data"
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    QS="spal-zhan"
    echo "Start $QS with $QUERY on $data"
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    
    # bso
    TOOL="bso";
    QUERY="bso-zhan";
    QS="bso-zhan";
    BSOCONF="lookDtst";  # for BSO only
    echo "Start $QS with $QUERY on $data"
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --lookDtst $BSOCONF --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
done
