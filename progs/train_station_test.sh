#!/bin/bash

# For testing locally before running on the cluster

parameter_file=$1
station_id=$2
data_dir_f=${3-${HOME}/data/DWL/data/ozdata_hrs}
src_dir=${4-${HOME}/src}

my_prog=${src_dir}/monthly-lstm-runoff/progs/train_tf.py

if [ ! -e ${my_prog} ]; then
    echo "FAILED: program not found: $my_prog";
    exit 1;
fi

if [ ! -e ${parameter_file} ]; then
    echo "FAILED: file not found: $parameter_file";
    exit 1;
fi

. ${parameter_file}

start_time=`date`

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


if [ ! -z ${PRINT_OUT_DLCMD+x} ]; then 
    echo python 
    echo ${my_prog}
    echo --datadir 
    echo ${data_dir_f}
    echo --outdir 
    echo ${out_dir_f}
    echo --seed
    echo 42
    echo --n_epochs 
    echo ${n_epochs}
    echo --train_start_date 
    echo ${train_start_date}
    echo --train_end_date 
    echo ${train_end_date}
    echo --eval_start_date 
    echo ${eval_start_date}
    echo --eval_end_date 
    echo ${eval_end_date}
    echo --model_id 
    echo ${model_id}
    echo --batch_size 
    echo ${batch_size}
    echo --seq_length 
    echo ${seq_length}
    echo --num_features 
    echo ${num_features}
    echo --feature_ids 
    echo ${feature_ids}
    echo --steps_per_epoch 
    echo ${steps_per_epoch}
    echo --stride 
    echo ${stride}
    echo --lstm_dim 
    echo ${lstm_dim}
    echo --log_dir 
    echo ${log_dir}
    echo --logging 
    echo ${logging}
    echo --early_stopping_patience 
    echo ${early_stopping_patience}
    echo --lr_patience 
    echo ${lr_patience}
    echo --lr_factor 
    echo ${lr_factor}
    echo --lr_start 
    echo ${lr_start}
    echo --dropout 
    echo ${dropout}
    echo --recurrent_dropout 
    echo ${recurrent_dropout}
    echo --runoff_scaling_option 
    echo ${runoff_scaling_option}
    echo ${bool_opts}
    echo ${station_id}
exit 0
fi

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
