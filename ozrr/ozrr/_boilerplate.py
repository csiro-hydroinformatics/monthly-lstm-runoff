"""A place to gather code that otherwise tends to repeat itself in exploratory notebooks 
"""

from pathlib import Path
import pandas as pd
import os

# TODO
# These were convenient to reduce entropy early on.
# Needs adaptation for third parties
# Revise when porting notebooks for a larger audience.

_PATH_SHARED_DRIVE = "/not_found"
_PATH_LOCAL_DIR = "/not_found"
_PATH_SHARED_MACHINE = "/not_found"

if "OZRR_PATH_SHARED_DRIVE" in os.environ: _PATH_SHARED_DRIVE = os.environ["OZRR_PATH_SHARED_DRIVE"]
if "OZRR_PATH_LOCAL_DIR" in os.environ: _PATH_LOCAL_DIR = os.environ["OZRR_PATH_LOCAL_DIR"]
if "OZRR_PATH_SHARED_MACHINE" in os.environ: _PATH_SHARED_MACHINE = os.environ["OZRR_PATH_SHARED_MACHINE"]

def create_trainer(data_repo, station_id, out_dir=None):
    from ozrr.tfmodels import CatchmentTraining, checked_mkdir, mk_model_filename, lstm_single, OUT_MODELS_DIRNAME
    if out_dir is not None:
        model_dir = checked_mkdir(out_dir / OUT_MODELS_DIRNAME)
        model_file = mk_model_filename(model_dir, station_id, with_timestamp=False)
    else:
        model_file = None
    ct = CatchmentTraining(out_dir, station_id, data_repo, model_file=model_file)
    # circa; let the data module do its magic to find the last full month of data prior to that for aggregation
    ct.conf.eval_end_date = pd.Timestamp("2020-07-15")
    ct.conf.seq_length = 12
    ct.conf.steps_per_epoch = 400
    ct.conf.lstm_dim = 40
    ct.conf.shuffle = True
    ct.conf.stride = 1
    ct.conf.batch_size = 24
    ct.conf.stateful = False
    ct.fit_verbosity = 1  # https://www.tensorflow.org/api_docs/python/tf/keras/Model can be string or integer.
    ct.eval_verbosity = 0  

    ct.conf.n_epochs = 60

    #we will use 4 out of 5 features in the data prepated: P, ET, mean TMax, and effective monthly rainfall.
    ct.conf.num_features = 4
    ct.conf.model_func = lstm_single
    return ct

def end_of_months_timestamps():
    run_start = pd.Timestamp("1950-01-31")
    # 2 year warmup, maybe. TBC.
    calib_start = run_start + pd.DateOffset(years=2)
    calib_end = pd.Timestamp("1995-12-31")
    # validation/verification period
    valid_start = pd.Timestamp("1996-01-31")
    valid_end = pd.Timestamp("2020-06-30")
    return run_start, calib_start, calib_end, valid_start, valid_end

class OzrrPathFinder:
    """Helper class to find paths to data across machines."""
    def __init__(self) -> None:
        pass

    def find_input_data_dir(self, must_exist:bool=True) -> Path:
        import os
        root_dir = Path(_PATH_SHARED_DRIVE)
        if not root_dir.exists():
            root_dir = Path(_PATH_LOCAL_DIR)
        if not root_dir.exists():
            root_dir = Path(_PATH_SHARED_MACHINE)
        if root_dir.exists():
            root_data_dir = root_dir / 'data'
            ozdata_dir = root_data_dir / 'ozdata_hrs'
        else:
            if 'AUSRR_DATA_DIR' in os.environ:
                root_dir_f = os.environ['AUSRR_DATA_DIR']
                ozdata_dir = Path(root_dir_f)
            else:
                raise FileNotFoundError("No default legacy input data found, and no env var AUSRR_DATA_DIR")
        if must_exist and not ozdata_dir.exists():
            raise FileNotFoundError(f"Directory {ozdata_dir} not found")
        return ozdata_dir

    def find_output_data_dir(self) -> Path:
        root_res_dir = Path(_PATH_SHARED_DRIVE)
        if not root_res_dir.exists():
            root_res_dir = Path(_PATH_SHARED_MACHINE)
        if root_res_dir.exists():
            root_data_dir = root_res_dir / 'data'
            results_out_dir = root_data_dir / "ozdata_hrs_out" 
        else:
            raise FileNotFoundError("No default suitable path to output data found on this machine")

        assert results_out_dir.exists()
        return results_out_dir

    def find_root_dir(self) -> Path:
        root_res_dir = Path(_PATH_SHARED_DRIVE)
        if not root_res_dir.exists():
            root_res_dir = Path(_PATH_SHARED_MACHINE)
        if not root_res_dir.exists():
            raise FileNotFoundError("No default suitable path to output data found on this machine")

        return root_res_dir

    def find_collated_results_ncfile(self) -> Path:
        results_out_dir = self.find_output_data_dir()
        return results_out_dir / "collated_results.nc"

    def find_auxiliary_dir(self) -> Path:
        rd = self.find_root_dir()
        return rd / "data" / "auxiliary"
