TASK="zhan";
# exp settings
SEED="0";
N_EXP="150";
# For multi-processing
N_JOBS="10";
END=$(($SEED + $N_EXP - 1));

# iris
data="iris";
n_init_lbl=30;
## google
TOOL="google";
QUERY="google-zhan";
for QS in "uniform-zhan" "hier-zhan" "margin-zhan" "graph-zhan" "infodiv-zhan" "mcm-zhan"; do
    echo "Start $QS with $HS on $data, repeat $N_EXP times."
    for s in $(seq $SEED $N_JOBS $END); do #
        python main.py --qs_name $QS --hs_name $QUERY --seed $s --data_set $data --init_lbl_size $n_init_lbl --tool $TOOL --gs_name $TASK --n_jobs $N_JOBS --n_trials $N_JOBS;
    done
done
## libact
TOOL="libact";
QUERY="qbc-zhan";
QS="qbc-zhan";
echo "Start $QS with $HS on $data, repeat $N_EXP times."
for s in $(seq $SEED $N_JOBS $END); do #
    python main.py --qs_name $QS --hs_name $QUERY --seed $s --data_set $data --init_lbl_size $n_init_lbl --tool $TOOL --gs_name $TASK --n_jobs $N_JOBS --n_trials $N_JOBS;
done

QUERY="libact-zhan";
QS="kcenter-zhan";
echo "Start $QS with $HS on $data, repeat $N_EXP times."
for s in $(seq $SEED $N_JOBS $END); do #
    python main.py --qs_name $QS --hs_name $QUERY --seed $s --data_set $data --init_lbl_size $n_init_lbl --tool $TOOL --gs_name $TASK --n_jobs $N_JOBS --n_trials $N_JOBS;
done
## alipy
TOOL="alipy";
QS="eer-zhan";
QUERY="eer-zhan";
echo "Start $QS with $HS on $data, repeat $N_EXP times."
for s in $(seq $SEED $N_JOBS $END); do #
    python main.py --qs_name $QS --hs_name $QUERY --seed $s --data_set $data --init_lbl_size $n_init_lbl --tool $TOOL --gs_name $TASK --n_jobs $N_JOBS --n_trials $N_JOBS;
done
## bso
TOOL="bso";  # "google" "libact" "alipy" "bso"
QS="bso-zhan";
QUERY="bso-zhan";
BSOCONF="lookDtst";  # for BSO only
echo "Start $QS with $HS on $data, repeat $N_EXP times."
for s in $(seq $SEED $N_JOBS $END); do #
    python main.py --qs_name $QS --hs_name $QUERY --seed $s --data_set $data --init_lbl_size $n_init_lbl --tool $TOOL --gs_name $TASK --n_jobs $N_JOBS --n_trials $N_JOBS --lookDtst $BSOCONF;
done
