"""Metrics for the model training and testing"""

from collections import OrderedDict
from pathlib import Path
from telnetlib import SE
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

import xarray as xr

from ozrr_data.data import as_ts_df, monthly_timedim, to_monthly_xr, to_xarray_ts

from ozrr_data.conventions import (
    LAT_VARNAME,
    LON_VARNAME,
    METRIC_DIM_NAME,
    MODEL_DIM_NAME,
    PERIOD_DIM_NAME,
    STATIONS_DIM_NAME,
    TESTING_LABEL,
    TRAINING_LABEL,
)

from ozrr_data.repository import get_catchment_id
from ozrr_data.read import load_bivariate_series
from ozrr_data.repository import OzDataProvider


def remove_nans(obs:np.ndarray, sim:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    tt = np.logical_not(np.isnan(obs))
    # we also need to remove nans from items in the simulation (neural net first few values are NaN by design)
    tt_sim = np.logical_not(np.isnan(sim))
    tt = np.logical_and(tt, tt_sim)
    obs_t = obs[tt]
    sim_t = sim[tt]
    return obs_t, sim_t

def reciprocal(x, epsilon=1.0):
    return 1 / (x + epsilon)

def mae(obs:np.ndarray, sim:np.ndarray) -> float:
    obs, sim = remove_nans(obs, sim)
    n = len(obs)
    if n == 0:
        return 0.0
    diff = sim - obs
    return np.sum(np.abs(diff)) / n

def mae_sqrt(obs:np.ndarray, sim:np.ndarray) -> float:
    obs = np.sqrt(obs)
    sim = np.sqrt(sim)
    return mae(obs, sim)

def nse(obs:np.ndarray, sim:np.ndarray) -> float:
    from hydrodiy.stat.metrics import nse as hnse
    return hnse(obs, sim, excludenull=True)

def recip_nse(obs:np.ndarray, sim:np.ndarray) -> float:
    return nse(reciprocal(obs), reciprocal(sim))

def bias(obs:np.ndarray, sim:np.ndarray) -> float:
    from hydrodiy.stat.metrics import bias as hbias
    return hbias(obs, sim, excludenull=True)

SQRT_MAE_METRIC_ID = "SqrtMAE"
MAE_METRIC_ID = "MAE"
NSE_METRIC_ID = "NSE"
RECIP_NSE_METRIC_ID = "RecipNSE"
BIAS_METRIC_ID = "Bias"

def default_metric_identifiers():
    return [SQRT_MAE_METRIC_ID,
        MAE_METRIC_ID,
        NSE_METRIC_ID,
        RECIP_NSE_METRIC_ID,
        BIAS_METRIC_ID]

DictBivariateStat = Dict[str, Callable[[np.ndarray, np.ndarray], float]]

def evaluation_metrics_funcs() -> DictBivariateStat:
    return OrderedDict({
        SQRT_MAE_METRIC_ID: mae_sqrt,
        MAE_METRIC_ID: mae,
        NSE_METRIC_ID: nse,
        RECIP_NSE_METRIC_ID: recip_nse,
        BIAS_METRIC_ID: bias,
    })

import pandas as pd

def bivariate_metrics(obs:np.ndarray, sim:np.ndarray, metric_funcs:DictBivariateStat, prefix=""):
    dd = {(prefix + k): v(obs, sim) for (k, v) in metric_funcs.items()}
    return dd

def get_metrics(series:pd.DataFrame, start:pd.Timestamp, end:pd.Timestamp, metric_funcs:DictBivariateStat, prefix="", obs_colname='Observed', mod_colname='Modelled'):
    x_calib = series[slice(start, end)]
    obs = x_calib[obs_colname].values
    sim = x_calib[mod_colname].values
    return bivariate_metrics(obs, sim, metric_funcs, prefix)


def find_metrics_names(d_wpb:pd.DataFrame, prefix="calib_"):
    cols = d_wpb.columns[[x.startswith(prefix) for x in d_wpb.columns]].values
    metric_names = [x.replace(prefix, "") for x in cols]
    return metric_names

def get_metrics_columns(d_wpb:pd.DataFrame, metric_names:Sequence[str], prefix="calib_"):
    cols = [prefix + x for x in metric_names]
    return d_wpb[cols]

def collate_metrics(list_metrics, met_keys):
    nn = np.arange(len(list_metrics))
    metrics = {k: [list_metrics[i][k] for i in nn] for k in met_keys}
    d_collated = metrics
    d = pd.DataFrame.from_dict(d_collated)
    return d

class MetricsCalculations():
    def __init__(self, metric_funcs:DictBivariateStat, calib_start, calib_end, valid_start, valid_end
    ) -> None:
        self.metric_funcs:DictBivariateStat = metric_funcs
        self.calib_start = calib_start
        self.calib_end = calib_end
        self.valid_start = valid_start
        self.valid_end = valid_end
        self.training_label = TRAINING_LABEL
        self.testing_label = TESTING_LABEL

    def get_metrics_from_tseries(self, sl_df_ts:Sequence[pd.DataFrame], station_ids:Sequence[str], met_keys, model_id):
        from tqdm import tqdm
        n = len(sl_df_ts)
        list_metrics_c = [
            get_metrics(sl_df_ts[i], self.calib_start, self.calib_end, self.metric_funcs, prefix="")
            for i in tqdm(range(n))
        ]
        list_metrics_t = [
            get_metrics(sl_df_ts[i], self.valid_start, self.valid_end, self.metric_funcs, prefix="")
            for i in tqdm(range(n))
        ]

        metrics_c = collate_metrics(list_metrics_c, met_keys)
        metrics_t = collate_metrics(list_metrics_t, met_keys)

        data = np.stack([metrics_c.values, metrics_t.values])
        data = np.expand_dims(data, -1)

        lastm_one_x = xr.DataArray(
            data,
            coords=[[self.training_label, self.testing_label], station_ids, met_keys, [model_id]],
            dims=[PERIOD_DIM_NAME, STATIONS_DIM_NAME, METRIC_DIM_NAME, MODEL_DIM_NAME],
        )
        return lastm_one_x

### Non-linear scaling used for color schemes in map visualisation.

def normalised_performance_comparison(metric_model_2:np.ndarray, metric_model_1:np.ndarray, max_sum:np.ndarray=0) -> float:
    sum_metric = metric_model_2 + metric_model_1
    denom = max_sum - sum_metric
    # np.array; just let np.nan happen if they do
    # if denom == 0:
    # 
    x = (metric_model_2 - metric_model_1) / denom
    # colormap in the chloropleth is not happy with nan
    nan_indx = np.argwhere(np.isnan(x))
    x[nan_indx] = 0.0 # HACK? not too bad. DL models have NaN on MAE metrics, somehow, sometimes.
    return x

# Brings a comparison of NSE scores (or anything for "higher is better"?) to a derived measure between -1 and +1
def nse_performance_comparison(nse_model_2, nse_model_1):
    return normalised_performance_comparison(nse_model_2, nse_model_1, 2)

def magnify_comparison(x: float):
    import math
    # Apply a transform to enhance the difference in colors for contrasting the picture.
    exponent = 0.2
    y = abs(x)
    z = math.pow(y, exponent)
    return z if x > 0 else -z


#################################
# Metrics I/O and collation
#################################



def _get_catchment_id_custom(f: Path):
    return f.name.split("_")[0]


from ozrr._boilerplate import end_of_months_timestamps
_run_start, _calib_start, _calib_end, _valid_start, _valid_end = end_of_months_timestamps()

def ingest_series_results(dl_results_dir: Path, model_id:str) -> Tuple[xr.DataArray, xr.DataArray]:
    """Loads from disk and into memory the time series of simulations, and calculate a set of metrics on these.

    Args:
        dl_results_dir (Path): Path to the root of the simulation results on disk
        model_id (str): model identifier, e.g. "lstm_single"

    Returns:
        Tuple[xr.DataArray, xr.DataArray]: xarray representations of (1) observed/simulated series, and (2) model metrics
    """
    sl_df, station_ids = load_bivariate_series(
        dl_results_dir, glob="*_series*.csv", get_catchment_id=_get_catchment_id_custom
    )
    sl_df_ts = [as_ts_df(x) for x in sl_df]
    metric_funcs = evaluation_metrics_funcs()
    met_keys = list(metric_funcs.keys())

    met_calc = MetricsCalculations(
        metric_funcs, _calib_start, _calib_end, _valid_start, _valid_end
    )

    model_metrics = met_calc.get_metrics_from_tseries(
        sl_df_ts, station_ids, met_keys, model_id
    )
    rr_series = to_monthly_xr(sl_df, station_ids)
    return rr_series, model_metrics



WAPABA_KEY = "WAPABA"

# WAPABA related functions:

def station_from_wapaba_fn(f:Path) -> str:
    """Gets the station identifier from a wapaba parameter set CSV file

    Args:
        f (Path): file path e.g. 'blah/wapaba/calibrated/WAPABA/params_410061.csv'

    Returns:
        str: station id e.g. '410061'
    """
    return f.name.split("_")[1].split(".")[0]

def sanitise_parameterset(d:pd.DataFrame, station_id:str) -> pd.DataFrame:
    """Sanitize parameter names to be short names, not FQNs such as 'catchment.subarea1.K'"""
    pvals = d.Value.values
    pnames = [x.split(".")[-1] for x in d.Name.values]
    dd = pd.DataFrame(pvals, pnames).T
    dd[STATIONS_DIM_NAME] = station_id
    return dd


def load_wapaba_stats(wapaba_csv_result_fn:str) -> Tuple[pd.DataFrame, List[str]]:
    """Loads wapaba collated metrics/statistics from a file"""
    d_wpb = pd.read_csv(wapaba_csv_result_fn)
    station_ids = [str(x) for x in d_wpb.station_id.values]
    return d_wpb, station_ids

def ingest_wapaba_stats(wapaba_dir:Path, d_wpb:pd.DataFrame, station_ids:Sequence[str]) -> xr.DataArray:
    """Ingest wapaba metrics and parameters into a dataarray for in-memory indexed access

    Args:
        wapaba_dir (Path): Parent directory with parameter files "param*.csv"
        d_wpb (pd.DataFrame): metrics loaded with `load_wapaba_stats`
        station_ids (Sequence[str]): Station ids

    Returns:
        xr.DataArray: 4D data cube of metrics: dimensions PERIOD_DIM_NAME, STATIONS_DIM_NAME, METRIC_DIM_NAME, MODEL_DIM_NAME
    """
    files_csv = [f for f in wapaba_dir.glob("param*.csv")]
    pp = [
        sanitise_parameterset(pd.read_csv(f), station_from_wapaba_fn(f)) for f in files_csv
    ]

    params = pd.concat(pp)
    params = params.set_index(STATIONS_DIM_NAME)

    met_n = find_metrics_names(d_wpb, prefix="calib_")

    mm_calib = get_metrics_columns(d_wpb, met_n, prefix="calib_")
    mm_test = get_metrics_columns(d_wpb, met_n, prefix="test_")

    data = np.stack([mm_calib.values, mm_test.values])
    data = np.expand_dims(data, -1)
    #data.shape

    model_id = WAPABA_KEY

    wpb_x = xr.DataArray(
        data,
        coords=[[TRAINING_LABEL, TESTING_LABEL], station_ids, met_n, [model_id]],
        dims=[PERIOD_DIM_NAME, STATIONS_DIM_NAME, METRIC_DIM_NAME, MODEL_DIM_NAME],
    )
    return wpb_x

def ingest_wapaba_series(wapaba_dir:Path):
    wpb_df, station_ids = load_bivariate_series(
        wapaba_dir, glob="series*.csv", get_catchment_id=get_catchment_id
    )
    wpb_xr = to_xarray_ts(wpb_df, station_ids)
    rr_series_wpb_xr = xr.concat(wpb_xr, pd.Index(station_ids, name=STATIONS_DIM_NAME))
    rr_series_wapaba = monthly_timedim(rr_series_wpb_xr)
    return rr_series_wapaba

class MetricsCollator:
    """Collates model predictions and metrics from experiments into xarray for easier data handling
    """    
    def __init__(self, 
        data_repo:OzDataProvider,
        model_results_loc:Dict[str,Path],
        wapaba_out_dir:Optional[Path]=None,
        ) -> None:
        """Define a metrics collator for several monthly rainfall-runoff experiments

        Args:
            data_repo (OzDataProvider): Input data repository
            model_results_loc (Dict[str,Path]): locations of . Keys are arbitraty experiment identifiers, values are paths with output csv time series such as '/blah/allresults/2022-11-14_2features_stdscaler/lstm_single/preds'
            wapaba_out_dir (Optional[Path], optional): Optional path to wapaba results if it has to be part of the collation. Defaults to None.
        """
        self.model_results_loc = model_results_loc
        self.data_repo = data_repo
        self.wapaba_out_dir = wapaba_out_dir
        self._ingested = None

    def load_cache(self, netcdf_fn:str):
        if self._ingested is not None:
            raise Exception("A dataset is already loaded in this collator. You may not load an existing dataset")
        prior_ds = MetricsCollator.load_netcdf(netcdf_fn)
        experiment_keys = set(prior_ds.model.values).intersection(list(self.model_results_loc.keys()))
        if len(experiment_keys) > 0:
            msg = ','.join(experiment_keys)
            raise Exception( f'There are overlapping experiment identifiers: {msg}')
        else:
            self._ingested = prior_ds

    def ingest(self):

        from tqdm import tqdm

        do_wabapa = self.wapaba_out_dir is not None
        if do_wabapa:
            wapaba_csv_result_fn = self.wapaba_out_dir / "collated_wapaba_metrics.csv"
            wapaba_dir = self.wapaba_out_dir / "calibrated" / WAPABA_KEY
            assert wapaba_csv_result_fn.exists()
            d_wpb, station_ids = load_wapaba_stats(wapaba_csv_result_fn)
            rr_series_wapaba = ingest_wapaba_series(wapaba_dir)
            wpb_x = ingest_wapaba_stats(wapaba_dir, d_wpb, station_ids)

        model_results = [
            ingest_series_results(v, k)
            for k, v in tqdm(self.model_results_loc.items())
        ]
        # (rr_series, model_metrics)
        rr_series_list = [x[0] for x in model_results]
        model_metrics_list = [x[1] for x in model_results]

        model_series = rr_series_list
        model_ids = list(self.model_results_loc.keys())

        if do_wabapa:
            model_series = [rr_series_wapaba] + model_series
            model_ids = [WAPABA_KEY] + model_ids
            model_metrics_list = [wpb_x] + model_metrics_list

        fitnesses = xr.concat(model_metrics_list, dim=MODEL_DIM_NAME)

        monthly_series = xr.concat(
            model_series,
            dim=pd.Index(model_ids, name=MODEL_DIM_NAME),
        )

        subset_stations = self.data_repo.data_for_station(monthly_series.station)
        monthly_series.coords[LAT_VARNAME] = (STATIONS_DIM_NAME, subset_stations.lat.data)
        monthly_series.coords[LON_VARNAME] = (STATIONS_DIM_NAME, subset_stations.lon.data)

        results_ds = xr.Dataset({"metrics": fitnesses, "series": monthly_series})

        if self._ingested is None:
            self._ingested = results_ds
        else:
            self._ingested = self._ingested.merge(results_ds)

    def append(self, model_results_locations:Dict):

        from tqdm import tqdm

        model_results = [
            ingest_series_results(v, k)
            for k, v in tqdm(model_results_locations.items())
        ]
        # (rr_series, model_metrics)
        rr_series_list = [x[0] for x in model_results]
        model_metrics_list = [x[1] for x in model_results]

        model_series = rr_series_list
        model_ids = list(model_results_locations.keys())

        fitnesses = xr.concat(model_metrics_list, dim=MODEL_DIM_NAME)

        monthly_series = xr.concat(
            model_series,
            dim=pd.Index(model_ids, name=MODEL_DIM_NAME),
        )

        subset_stations = self.data_repo.data_for_station(monthly_series.station)
        monthly_series.coords[LAT_VARNAME] = (STATIONS_DIM_NAME, subset_stations.lat.data)
        monthly_series.coords[LON_VARNAME] = (STATIONS_DIM_NAME, subset_stations.lon.data)

        results_ds = xr.Dataset({"metrics": fitnesses, "series": monthly_series})

        if self._ingested is None:
            self._ingested = results_ds
        else:
            self._ingested = self._ingested.merge(results_ds)

    @property
    def collated_dataset(self):
        return self._ingested

    def save(self, out_fn:str):
        """Saves the collated dataset to disk"""
        import os

        try:
            self._ingested.to_netcdf(out_fn, mode="w")
        except Exception as e:
            os.remove(out_fn)
            raise e

    @staticmethod
    def load_netcdf(in_fn:str) -> xr.Dataset:
        """Loads the collated dataset from a netcdf file on disk"""
        return xr.load_dataset(in_fn)

