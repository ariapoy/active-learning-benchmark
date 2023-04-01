TASK="zhan";
s="0";
N_EXP="1";
N_JOBS="1";
for data in "spambase" "banana" "phoneme" "ringnorm" "twonorm" "phishing"; do
    # libact
    TOOL="libact";
    QUERY="libact-zhan";
    for QS in "quire-zhan"; do
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
    QS="bmdr-zhan"
    echo "Start $QS with $QUERY on $data"
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    QS="spal-zhan"
    echo "Start $QS with $QUERY on $data"
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;
    
    bso
    TOOL="bso";
    QUERY="bso-zhan";
    QS="bso-zhan";
    BSOCONF="lookDtst";  # for BSO only
    echo "Start $QS with $QUERY on $data"
    timeout 259200 python main.py --tool $TOOL --qs_name $QS --lookDtst $BSOCONF --hs_name $QUERY --gs_name $TASK --seed $s --n_jobs $N_JOBS --n_trials $N_JOBS --data_set $data;

done
