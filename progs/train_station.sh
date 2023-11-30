#!/bin/bash

module load python

env_name=ozrr_mycluster
work_dir=/datasets/work/path/to/my/workdir
src_dir=${work_dir}/src_pub
my_prog=${src_dir}/monthly-lstm-runoff/progs/train_tf.py

venv_dir=${work_dir}/venv/${env_name}
source ${venv_dir}/bin/activate

data_dir_f=/datasets/work/path/to/my/workdir/data/ozdata_hrs

parameter_file=$1

if [ ! -e ${parameter_file} ]; then
    echo "FAILED: file not found: $parameter_file";
    exit 1;
fi

. ${parameter_file}

station_id=$2


# Cater for boolean options. https://docs.python.org/3/library/argparse.html?highlight=booleanoptionalaction
_parse_bool () {
    x=$1
    opt=$2
    y="`echo "$x" | awk '{print tolower($0)}'`"
    if [ "$y" = "true" ]; then
        s="--${opt}";
    elif [ "$y" = "false" ]; then
        s="--no-${opt}";
    else
        s="";
    fi
    echo $s
}

shuffle_opt=`_parse_bool "${shuffle}" shuffle`
use_validation_opt=`_parse_bool "${use_validation}" use_validation`
early_stopping_opt=`_parse_bool "${early_stopping}" early_stopping`
reproducible_initialisers_opt=`_parse_bool "${reproducible_initialisers}" reproducible_initialisers`

bool_opts=" ${shuffle_opt} ${use_validation_opt} ${early_stopping_opt} ${reproducible_initialisers_opt} "


start_time=`date`

python ${my_prog} --datadir ${data_dir_f} \
    --outdir ${out_dir_f} \
    --seed 42 \
    --n_epochs ${n_epochs} \
    --train_start_date ${train_start_date} \
    --train_end_date ${train_end_date} \
    --eval_start_date ${eval_start_date} \
    --eval_end_date ${eval_end_date} \
    --model_id ${model_id} \
    --batch_size ${batch_size} \
    --seq_length ${seq_length} \
    --num_features ${num_features} \
    --feature_ids ${feature_ids} \
    --steps_per_epoch ${steps_per_epoch} \
    --stride ${stride} \
    --lstm_dim ${lstm_dim} \
    --log_dir ${log_dir} \
    --early_stopping_patience ${early_stopping_patience} \
    --lr_patience ${lr_patience} \
    --lr_factor ${lr_factor} \
    --lr_start ${lr_start} \
    --dropout ${dropout} \
    --recurrent_dropout ${recurrent_dropout} \
    --runoff_scaling_option ${runoff_scaling_option} \
    ${bool_opts} \
    ${station_id}

end_time=`date`

# usage: train_tf.py [OPTION] [STATON_ID]

# Train a catchment using deep learning on tensorflow

# positional arguments:
#   station_id            one or more station identifiers

# optional arguments:
#   -h, --help            show this help message and exit
#   -v, --version         show program's version number and exit
#   -d DATADIR, --datadir DATADIR
#                         top directory of the input data
#   -o OUTDIR, --outdir OUTDIR
#                         top directory of the output data
#   -s SEED, --seed SEED  seed for the training process
#   --train_start_date TRAIN_START_DATE
#                         train_start_date
#   --train_end_date TRAIN_END_DATE
#                         train_end_date
#   --eval_start_date EVAL_START_DATE
#                         eval_start_date
#   --eval_end_date EVAL_END_DATE
#                         eval_end_date
#   --batch_size BATCH_SIZE
#                         batch_size
#   --seq_length SEQ_LENGTH
#                         seq_length
#   --num_features NUM_FEATURES
#                         num_features
#   --n_epochs N_EPOCHS   n_epochs
#   --steps_per_epoch STEPS_PER_EPOCH
#                         steps_per_epoch
#   --stride STRIDE       stride
#   --lstm_dim LSTM_DIM   lstm_dim


echo ####################################################################
echo "Station ${station_id}: started: ${start_time} finished: ${end_time}"
echo ####################################################################
