import argparse
from collections import OrderedDict

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="python %(prog)s [OPTIONS] [STATION_ID] \n       python %(prog)s --help for details",
        description="Train a catchment using deep learning on tensorflow",
        add_help=True
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 1.0.0"
    )
    parser.add_argument(
        "-d", "--datadir", default="/home/xxxyyy/data/DWL/monthly",
        type=str,
        help='top directory of the input data')
    parser.add_argument(
        "-o", "--outdir", default="/home/xxxyyy/data/DWL/monthly_out",
        type=str,
        help='top directory of the output data')
    parser.add_argument(
        "-s", "--seed", default=42,
        type=int,
        help='seed for the training process')
    parser.add_argument("--train_start_date", default="1950-01-01", type=str, help="train_start_date")
    parser.add_argument("--train_end_date", default="1995-12-31", type=str, help="train_end_date")
    parser.add_argument("--eval_start_date", default="1996-01-01", type=str, help="eval_start_date")
    parser.add_argument("--eval_end_date", default="2009-12-31", type=str, help="eval_end_date")
    parser.add_argument("--model_id", default="lstm_single", type=str, help="model identifier; one of: lstm_single, lstm_two_layers, lstm_enc_dec, cn_lstm, multi_sequence_lstm, seq2seq, cn_multi_lstm, four_layer_lstm, e2d2, fcn, multi_head_lstm, DC_CNN_Model, cn_parallel_lstm, ts_unet")
    parser.add_argument("--batch_size", default = 32, type=int, help="batch_size")
    parser.add_argument("--seq_length", default = 24, type=int, help="seq_length")
    parser.add_argument("--num_features", default = 2, type=int, help="num_features")
    parser.add_argument("--feature_ids", type=str, nargs='+', help="feature_ids") # Optional added 202301
    parser.add_argument("--n_epochs", default = 30, type=int, help="n_epochs")  # Number of training epochs
    parser.add_argument("--steps_per_epoch", default = 800, type=int, help="steps_per_epoch")
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default = True)
    parser.add_argument("--stride", default = 8, type=int, help="stride")
    parser.add_argument("--lstm_dim", default = 20, type=int, help="lstm_dim")
    parser.add_argument("--fit_verbosity", default = 0, type=int, help="fitting verbosity; should be 0 in batch mode on a cluster to avoid clogging the logs.")
    parser.add_argument("--eval_verbosity", default = 0, type=int, help="evaluation verbosity; should be 0 in batch mode on a cluster to avoid clogging the logs.")
    parser.add_argument("--logging", default="false", type=str, help="false (default), tensorboard, wandb")
    parser.add_argument("--log_dir", default="~/tmp/tb", type=str, help="directory for log output")
    parser.add_argument("--use_validation", default=False, action=argparse.BooleanOptionalAction,  help="Should a training/validation split be used for training")
    parser.add_argument("--early_stopping", default=True, action=argparse.BooleanOptionalAction,  help="Should a early stopping be used on the training or validation loss (latter if use_validation is true)")
    parser.add_argument("--early_stopping_patience", default = 5, type=int, help="Early stopping patience, if early_stopping is true")
    parser.add_argument("--lr_patience", default = 3, type=int, help="Patience (nb epoch) before applying the learning rate decay factor")
    parser.add_argument("--lr_factor", default = 0.1, type=float, help="Learning rate decay factor once LR patience has been reached")
    parser.add_argument("--lr_start", default = 0.0001, type=float, help="Starting learning rate")
    parser.add_argument("--dropout", default = 0.0, type=float, help="dropout factor (0 to 1)")
    parser.add_argument("--recurrent_dropout", default = 0.0, type=float, help="RNN dropout factor (0 to 1)")
    parser.add_argument("--runoff_scaling_option", default="std", type=str, help="scaling option for transformed runoff: std or minmax")
    parser.add_argument("--reproducible_initialisers", default=True, action=argparse.BooleanOptionalAction,  help="If true, LSTM layers weights are set deterministically, otherwise non-reproducible. Option only for backward compat.")

    parser.add_argument('station_id', type=str, nargs='+', help="one or more station identifiers")
    return parser

def _find_feature_ids(args):
    x:str = None
    if hasattr(args, "feature_ids"):
        x = args.feature_ids
        if isinstance(x, str):
            x = x.split(',')
    return x

def main() -> None:
    import pandas as pd
    parser = init_argparse()
    args = parser.parse_args()
    if not args.station_id:
        raise ValueError("station_id argument not found")
    root_dir_f = args.datadir
    out_dir_f = args.outdir
    
    feature_ids = _find_feature_ids(args)
    num_features = args.num_features
    if feature_ids is not None:
        num_features = len(feature_ids)

    kwargs = OrderedDict()

    kwargs["train_start_date"] = pd.Timestamp(args.train_start_date)
    kwargs["train_end_date"] = pd.Timestamp(args.train_end_date)
    kwargs["eval_start_date"] = pd.Timestamp(args.eval_start_date)
    kwargs["eval_end_date"] = pd.Timestamp(args.eval_end_date)
    kwargs["batch_size"] = args.batch_size
    kwargs["seq_length"] = args.seq_length
    kwargs["num_features"] = num_features
    kwargs["feature_ids"] = feature_ids
    kwargs["n_epochs"] = args.n_epochs
    kwargs["steps_per_epoch"] = args.steps_per_epoch
    # kwargs[ # "shuffle"] = args.shuffle
    kwargs["stride"] = args.stride
    kwargs["lstm_dim"] = args.lstm_dim
    kwargs["fit_verbosity"] = args.fit_verbosity
    kwargs["eval_verbosity"] = args.eval_verbosity
    kwargs["logging"] = args.logging
    kwargs["log_dir"] = args.log_dir
    kwargs["use_validation"] = args.use_validation
    kwargs["early_stopping"] = args.early_stopping
    kwargs["early_stopping_patience"] = args.early_stopping_patience
    kwargs["lr_patience"] = args.lr_patience
    kwargs["lr_factor"] = args.lr_factor
    kwargs["lr_start"] = args.lr_start
    kwargs["dropout"] = args.dropout
    kwargs["recurrent_dropout"] = args.recurrent_dropout
    kwargs["runoff_scaling_option"] = args.runoff_scaling_option
    kwargs["reproducible_initialisers"] = args.reproducible_initialisers

    # HACK: due to the implementation of the property model_func on the configuration object, this MUST be set last
    # See https://jira.csiro.au/browse/HYDROML-20
    kwargs["model_func"] = args.model_id # note slight difference

    # print(kwargs)

    seed = args.seed
    from time import sleep
    import numpy as np
    for station_id in args.station_id:
    #try:
        # print("do_learning({}, {}, {}, {})".format(station_id, root_dir_f, out_dir_f, n_epochs))
        # sleep(0.2 * np.random.randint(10))
        do_learning(station_id, root_dir_f, out_dir_f, seed, **kwargs)
    #except (FileNotFoundError, IsADirectoryError) as err:
    #    print(f"{sys.argv[0]}: {station_id}: {err.strerror}", file=sys.stderr)


# root_dir_f = "/home/xxxyyy/data/DWL/monthly"
# out_dir_f = "/home/xxxyyy/data/DWL/monthly_out"
# root_dir_f = "/home/kubeflow/datasets/lw-maas/work/xxxyyy/data/DWL/monthly"
# out_dir_f = "/home/kubeflow/datasets/lw-maas/work/xxxyyy/data/DWL/monthly_out"

# root_dir_f = "/datasets/work/path/to/my/workdir/data/ozdata_hrs"
# out_dir_f = "/datasets/work/path/to/my/workdir/data/ozdata_hrs_out"

def do_learning(station_id, root_dir_f, out_dir_f, seed=42, **kwargs):
    from pathlib import Path 
    # **NOTE** this training runs faster on a laptop CPU than a fairly decent GPU, about three times faster.
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU') # force CPU execution
    from ozrr_data.repository import load_aus_rr_data
    # from swift2.proto import PbmCalibrationBuilder, PbmModelFactory
    # import etu.batch as eb
    from ozrr.tfmodels import CatchmentTraining, checked_mkdir, mk_model_filename
    # root_dir = Path(root_dir_f)
    out_dir = Path(out_dir_f)
    data_repo = load_aus_rr_data(root_dir_f)
    ## GLOBAL SEED ##
    tf.random.set_seed(seed + 123)
    model_dir = checked_mkdir(out_dir / "models")
    model_file = mk_model_filename(model_dir, station_id, with_timestamp=False)
    ct = CatchmentTraining(out_dir, station_id, data_repo, model_file=model_file)
    # ct = CatchmentTraining(out_dir, station_id, data_repo, model_file=Path("/home/xxxyyy/data/DWL/monthly/out/models/410061_2022-01-28-14-42.hdf5"))
    # batch training: dont bloat the slurm output log:
    ct.fit_verbosity = 0 # int(kwargs["fit_verbosity"]) # https://www.tensorflow.org/api_docs/python/tf/keras/Model
    ct.eval_verbosity = 0 # int(kwargs["eval_verbosity"]) # https://www.tensorflow.org/api_docs/python/tf/keras/Model
    kwargs.pop("fit_verbosity")
    kwargs.pop("eval_verbosity")
    for key in kwargs:
        setattr(ct.conf, key, kwargs[key])
    ct.train(random_seed=seed)
    # my_cluster:
    # NotImplementedError: Cannot convert a symbolic Tensor (lstm/strided_slice:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported
    # May be an issue with the version of numpy:
    #  np.__file__
    # '/apps/python/3.9.4/lib/python3.9/site-packages/numpy-1.20.3-py3.9-linux-x86_64.egg/numpy/__init__.py'
    _ = ct.evaluate()
    # save predictions
    ct.generate_report()
    ct.sync_predictions()

if __name__ == "__main__":
    main()

