"""Deep Learning Models implemented with tensorflow"""

from datetime import datetime
from typing import TYPE_CHECKING, Callable, Generator, List, Optional, Sequence, Tuple, Union
from ozrr_data.conventions import EVAP_D_COL, RAIN_D_COL, RUNOFF_D_COL, TEMPMAX_D_COL
from sklearn.pipeline import Pipeline
import xarray as xr
import numpy as np
import pandas as pd


from pathlib import Path

from ozrr_data.conventions import TIME_DIM_NAME

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin
    TimeSeriesLike = Union[pd.Series, pd.DataFrame, xr.DataArray]

import tensorflow as tf
keras = tf.keras

from keras.models import Model
from keras.layers import (
    LSTM,
    Input,
    Dense,
)

from keras.regularizers import l2
# Side note - had issues importing TruncatedNormal in VSCode
# with from keras.initializers. Odd. Perhaps a Pylance issue. 
from keras.initializers.initializers_v2 import TruncatedNormal
from keras.optimizers import Adam
from ozrr.tfmetrics import nse, mean_absolute_error_na, remove_items_missing_observations

from ozrr_data.repository import (
    MaskTimeSeries,
    OzDataProvider,
)
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    FunctionTransformer,
)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import tensorflow as tf
keras = tf.keras
from keras import initializers

################################################
# Named constants
################################################

OBS_KEY = "obs"
PRED_KEY = "pred"
MM_P_MTH = "mm/mth"

OUT_PREDS_DIRNAME = "preds"
OUT_METRICS_DIRNAME = "metrics"
OUT_MODELS_DIRNAME = "models"
OUT_PLOTS_DIRNAME = "plots"

MM_TO_METRE = 1e-3

TANH_ACTIVATION = "tanh"
SAME_PAD = "same"
CAUSAL_PAD = "causal"

ELU_ACTIVATION = "elu"
RELU_ACTIVATION = "relu"
SOFTMAX_ACTIVATION = "softmax"
LINEAR_ACTIVATION = "linear"
K_INIT = "he_normal"

FEATURE_ID_TIME = "t"
FEATURE_ID_RAIN = "rain"
FEATURE_ID_PET = "pet"
FEATURE_ID_TMAX = "tmax"
FEATURE_ID_EFF_RAIN = "eff_rain"
FEATURE_ID_RAIN_SURPLUS = "rain_surplus"
FEATURE_ID_RUNOFF = "runoff"

# We use simple globals to handle how we run the trialled TF models
# `run_eagerly` was useful to decypher something in TF, but normally not of interest
run_eagerly=False
learning_rate=0.1

def _do_default_compile(model, compile_model:Optional[bool]=True):
    global run_eagerly
    global learning_rate
    if compile_model:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=mean_absolute_error_na, metrics=[nse], run_eagerly=run_eagerly)


# A simple 2 layer lstm model
# ref?
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/
def lstm_two_layers(
    pretrained_weights:Optional[Union[bool,str]]=False,
    lstm_dim:int=20,
    stateful:bool=False,
    batch_shape:Tuple=(32, 360, 2),
    input_size:Tuple=(100, 2),
    seed_init:Optional[int]=None,
    compile_model:Optional[bool]=True,
) -> Model:
    
    input1 = Input(batch_shape=batch_shape)
    ls1 = LSTM(
        lstm_dim,
        activation=TANH_ACTIVATION,
        return_sequences=True,
        stateful=stateful,
        recurrent_dropout=0.1,
    )(input1)
    ls2 = LSTM(lstm_dim, activation=TANH_ACTIVATION, stateful=stateful, recurrent_dropout=0.1)(
        ls1
    )
    ts1 = Dense(1)(ls2)
    model = Model(inputs=input1, outputs=ts1)
    _do_default_compile(model, compile_model)
    # model.summary()  # commented out to be less verbose
    # load pre-trained weights
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model

class LayerInitFactory:
    """Helper to initialise layer weights
    
    Tensorflow, and other frameworks, do not have a 
    deterministic reproducible behavior by default, 
    a puzzling choice IMO. This class helps to achieve a 
    random but reproducible outcome, all other things being equal."""
    def __init__(self, seed:Optional[int]) -> None:
        """Create a new factory to produce layer initialisers

        Args:
            seed (Optional[int]): The random seed. If None (default), behavior not reproducible (TF default behavior)
        """        
        self.seed = seed
        # We use a counter so that initialisers created in sequence are seeded predictably but not identically
        # See: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
        # multiple initializers will produce the same sequence when constructed with the same seed value. 
        if seed is not None:
            self._last_seed = seed + 1
        else:
            self._last_seed = None

    def init_k(self):
        """Creates a new `GlorotUniform` object for kernel initialisation (keras LSTM)"""
        l_init = tf.keras.initializers.GlorotUniform(seed=self._last_seed)
        self._up_seed()
        return l_init

    def _up_seed(self):
        if self._last_seed is not None:
            self._last_seed = self._last_seed + 1

    def init_rec(self):
        """Creates a new `Orthogonal` object for recurrent initialisation (keras LSTM)"""
        l_init = tf.keras.initializers.Orthogonal(seed=self._last_seed)
        self._up_seed()
        return l_init

def lstm_single(
    pretrained_weights:Optional[Union[bool,str]]=False,
    lstm_dim:int=20,
    stateful:bool=False,
    batch_shape:Tuple=(32, 12, 3),
    input_size:Tuple=(100, 2),
    seed_init:Optional[int]=None,
    compile_model:Optional[bool]=True,
):
    input1 = Input(batch_shape=batch_shape)

    f = LayerInitFactory(seed_init)
    ls1 = LSTM(lstm_dim,
        kernel_initializer=f.init_k(), recurrent_initializer=f.init_rec(), 
        activation=TANH_ACTIVATION, return_sequences=False, stateful=stateful)(
        input1
    )

    ts1 = Dense(1, kernel_initializer=f.init_k())(ls1)

    model = Model(inputs=input1, outputs=ts1)

    _do_default_compile(model, compile_model)

    # model.summary()  # commented out to be less verbose

    # load pre-trained weights
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model


class ModelFactory:
    """A factory to create models ready to fit. Helps repeatibility and hyperparameter search"""
    def __init__(self, learning_rate = 0.1, dropout = 0.1, recurrent_dropout = 0.1, run_eagerly = False, reproducible_initialisers = True
        ) -> None:
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.learning_rate = learning_rate
        self.run_eagerly = run_eagerly
        self.reproducible_initialisers = reproducible_initialisers

    def lstm_single(self,
        pretrained_weights:Optional[Union[bool,str]]=False,
        lstm_dim:int=20,
        stateful:bool=False,
        batch_shape:Tuple=(32, 12, 3),
        input_size:Tuple=(100, 2),
        seed_init:Optional[int]=None,
        compile_model:Optional[bool]=True,
    ):
        input1 = Input(batch_shape=batch_shape)

        # NOTE:
        # See https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
        # I tried to reverse engineer best I could to figure out the defaults kernel 
        # and recurrence initialisation, but this is not unambiguously documented.
        if self.reproducible_initialisers:
            f = LayerInitFactory(seed_init)
            ls1 = LSTM(lstm_dim,
                kernel_initializer=f.init_k(), recurrent_initializer=f.init_rec(), 
                activation=TANH_ACTIVATION, return_sequences=False, stateful=stateful, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)(
                input1
            )
            ts1 = Dense(1, kernel_initializer=f.init_k())(ls1)
        else:
            ls1 = LSTM(lstm_dim,
                activation=TANH_ACTIVATION, return_sequences=False, stateful=stateful, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)(
                input1
            )
            ts1 = Dense(1)(ls1)

        model = Model(inputs=input1, outputs=ts1)

        if compile_model:
            self._compile(model)

        # model.summary()  # commented out to be less verbose

        # load pre-trained weights
        if pretrained_weights:
            model.load_weights(pretrained_weights)
        return model


    def _compile(self, model:tf.keras.Model):
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=mean_absolute_error_na, metrics=[nse], run_eagerly=self.run_eagerly)


_named_models = {
    "lstm_single": lstm_single,
    "lstm_two_layers": lstm_two_layers,
    # "ts_unet": ts_unet,
}


def named_model(model_name):
    global _named_models
    return _named_models[model_name]


class Data:
    """Hold training data together"""

    def __init__(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        rain: pd.Series,
        pet: pd.Series,
        runoff: pd.Series,
        runoff_scaler: "TransformerMixin",
    ):
        self.rain = rain
        assert isinstance(pet, pd.Series)
        assert isinstance(runoff, pd.Series)
        assert isinstance(rain, pd.Series)
        self.pet = pet
        self.runoff = runoff
        self.runoff_scaler = runoff_scaler
        self.x = x
        self.y = y
        # NOTE: having these here is now very confusing after changes to the data scaler training methodology. Not the right design.
        self.obs_train: pd.Series = None
        self.pred_train: pd.Series = None
        self.obs_eval: pd.Series = None
        self.pred_eval: pd.Series = None
        self.eval_nse: float = np.nan
        self.train_nse: float = np.nan
        self.eval_bias: float = np.nan
        self.train_bias: float = np.nan


def fit_scaler(x: "TimeSeriesLike", x_scaler: "TransformerMixin"):
    x_scaled = x.values.reshape((-1, 1))
    x_scaler.fit(x_scaled)
    return x_scaler


def scale_data(x: "TimeSeriesLike", x_scaler: "TransformerMixin"):
    x_scaled = x.values.reshape((-1, 1))
    x_scaled = x_scaler.transform(
        x_scaled,
    )
    if isinstance(x, xr.DataArray):
        index = x.time.values
    else:
        index = x.index
    x_scaled = pd.Series(np.ndarray.flatten(x_scaled), index=index)
    # x_scaled = x_scaled.interpolate() # WARNING: NO NO. I think I had inherited this, which really is not acceptable for runoff.
    return x_scaled


class BatchProvider:
    """An iterator class that creates data batches according to a sampling strategy
    """    
    def __init__(
        self,
        x_samples: np.ndarray,
        y_samples: np.ndarray,
        num_samples: int,
        # num_features: int,
        stride: int,
        batch_num: int,
        shuffle: bool = True,
        random_seed: Optional[int] = None,
    ):
        import numpy as np
        self.num_samples = num_samples
        self.tailptr = batch_num  # random.randint(0, len(x_samples) - num_samples - 1) if shuffle else batch_num
        self.headptr = self.tailptr + num_samples
        # self.num_features = num_features
        self.stride = stride
        self.x_samples = x_samples.astype(np.float32) # NOTE: hunch this may help with https://jira.csiro.au/browse/HYDROML-19. Just a hunch.
        self.y_samples = y_samples.astype(np.float32)
        self.shuffle = shuffle
        self.batch_num = batch_num
        self.rng = np.random.RandomState(random_seed)

    def __iter__(self):
        return self

    def _nb_features(self) -> int:
        return self.x_samples.shape[1]

    def __next__(self):

        # Shuffle will pick a random num_samples from x_samples
        MAX_TRIAL_NONAN = 30
        y_isempty = True
        trials = 0
        while y_isempty:
            if self.shuffle:
                self.tailptr = self.rng.randint(
                    0, len(self.x_samples) - self.num_samples - 1
                )

                self.headptr = self.tailptr + self.num_samples

            # Load and shape return X array
            x = [ np.array(self.x_samples[self.tailptr : self.headptr, i]) for i in range(0, self._nb_features())]
            x = np.asarray(x)
            x = x.T

            # load the labels or groundtruth
            y = np.array(self.y_samples[self.headptr-1 : self.headptr])

            if not self.shuffle:
                self.headptr = self.headptr + self.stride
                self.tailptr = self.tailptr + self.stride

                # reset
                if self.headptr > len(self.x_samples) - 1:
                    self.tailptr = self.batch_num
                    self.headptr = self.tailptr + self.num_samples
                # we do not want to do trials (yet?) if we do not shuffle the sampling. Just let it go.
                break

            y_isempty = np.all(np.isnan(y))
            if not y_isempty:
                break
            trials = trials + 1
            if trials >= MAX_TRIAL_NONAN:
                break

        return x, y


def train_generator_batch(
    x_samples:np.ndarray,
    y_samples:np.ndarray,
    batch_size=2,
    num_samples=360,
    # num_features=2,
    stride=1,
    shuffle=False,
    random_seed: Optional[int] = None,
):
    # Create batch providers
    # from etu.batch import BatchProvider
    rng = np.random.RandomState(random_seed)
    large_ish = int(2 ** 30 - 1)
    providers = []
    for i in range(0, batch_size):
        providers.append(
            BatchProvider(
                x_samples,
                y_samples,
                num_samples,
                stride,
                i,
                shuffle=shuffle,
                random_seed=rng.randint(large_ish),
            )
        )

    while True:
        X, Y = np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        for provider in providers:
            x, y = next(provider)
            X = np.append(X, x)
            Y = np.append(Y, y)

        # reshape to (batchsize, numsamples, numfeatures)
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        Y = tf.convert_to_tensor(Y, dtype=tf.float32)
        num_features = x_samples.shape[1]
        X = tf.reshape(X, [batch_size, num_samples, num_features])
        Y = tf.reshape(Y, [batch_size, 1, 1])

        yield (X, Y)


class DataTransformParameters:
    """Manages the transformations ("scaling") of inputs/observations fed into the neural network"""

    def __init__(self,
        pet: "TimeSeriesLike",
        rain: "TimeSeriesLike",
        tmax: "TimeSeriesLike",
        eff_rain: "TimeSeriesLike",
        rain_surplus: "TimeSeriesLike",
        runoff: "TimeSeriesLike",
        second_runoff_scaler:Optional["TransformerMixin"] = None
        ) -> None:

        # Scale Data
        self.pet_scaler = StandardScaler()
        self.rain_scaler = StandardScaler()
        self.tmax_scaler = StandardScaler()
        self.rain_surplus_scaler = StandardScaler()
        self.eff_rain_scaler = StandardScaler()
        self.runoff_scaler_1 = FunctionTransformer(func=np.sqrt, inverse_func=np.square)
        self.runoff_scaler_2 = StandardScaler() if second_runoff_scaler is None else second_runoff_scaler
        # MinMaxScaler(feature_range=(-1.0, 1.0))

        _ = fit_scaler(pet, self.pet_scaler)
        _ = fit_scaler(rain, self.rain_scaler)
        _ = fit_scaler(tmax, self.tmax_scaler)
        _ = fit_scaler(eff_rain, self.eff_rain_scaler)
        _ = fit_scaler(rain_surplus, self.rain_surplus_scaler)

        runoff_sqrt = scale_data(runoff, self.runoff_scaler_1)
        _ = fit_scaler(runoff_sqrt, self.runoff_scaler_2)

        # self.runoff_scaler = self.runoff_scaler_1
        self.runoff_scaler = Pipeline(
            [("Sqrt", self.runoff_scaler_1), 
             ("TransformedRunoffScaler", self.runoff_scaler_2)]
        )

    def scale(self,
            pet,
            rain,
            tmax,
            eff_rain,
            rain_surplus,
            runoff):
        pet_scaled = scale_data(pet, self.pet_scaler)
        rain_scaled = scale_data(rain, self.rain_scaler)
        tmax_scaled = scale_data(tmax, self.tmax_scaler)
        eff_rain_scaled = scale_data(eff_rain, self.eff_rain_scaler)
        rain_surplus_scaled = scale_data(rain_surplus, self.rain_surplus_scaler)
        runoff_scaled = scale_data(runoff, self.runoff_scaler)

        return (pet_scaled,
                rain_scaled,
                tmax_scaled,
                eff_rain_scaled,
                rain_surplus_scaled,
                runoff_scaled)


EARLY_STOPPING_PATIENCE=5
REDUCE_LR_FACTOR=0.3
REDUCE_LR_PATIENCE=3

STD_SCALER='std'
MINMAX_SCALER='minmax'

class CatchmentTrainingConfig:
    def __init__(
        self,
        station_id,
        model_file,
        train_start_date="1950-01-01",
        train_end_date="1995-12-31",
        eval_start_date="1996-01-01",
        eval_end_date="2009-12-31",
    ):
        self.batch_size = 64
        self.seq_length = 365
        self.num_features = 2  # PET and Precipitation
        self.feature_ids = None
        self.n_epochs = 30  # Number of training epochs
        self.steps_per_epoch = 500
        self.validation_steps = 40
        self.shuffle = True
        self.stride = 8
        self.lstm_dim = 20
        self.train_start_date = pd.Timestamp(train_start_date)
        self.train_end_date = pd.Timestamp(train_end_date)
        self.eval_start_date = pd.Timestamp(eval_start_date)
        self.eval_end_date = pd.Timestamp(eval_end_date)
        self.station_id = station_id
        self.model_file = model_file
        self.stateful = False
        self._model_func = lstm_single
        self.use_validation = False 
        """Should there be a split of the training data into train/valid to probe for overfitting?"""
        self.early_stopping_patience = EARLY_STOPPING_PATIENCE
        self.lr_factor=REDUCE_LR_FACTOR
        self.lr_patience=REDUCE_LR_PATIENCE
        self.lr_start = 0.0001 # legacy. lrfinder suggests ~0.1
        self.early_stopping = True
        self.dropout = 0.0
        self.recurrent_dropout = 0.0

        self.logging = "false"
        self.log_dir: Optional[str] = None

        self.trained_scaler = None

        self.seed_init = 421

        self.runoff_scaling_option = STD_SCALER
        self.reproducible_initialisers = True

    @property
    def model_func(self):
        """gets/sets the function that creates a model"""
        return self._model_func

    @model_func.setter
    def model_func(self, val):
        if isinstance(val, str):
            self._model_func = self.named_model(val)
        else:
            self._model_func = val

    def named_model(self, model_id) -> Callable:
        if model_id == "lstm_single":
            self._mf = ModelFactory(self.lr_start, self.dropout, self.recurrent_dropout, reproducible_initialisers=self.reproducible_initialisers)
            return self._mf.lstm_single
        else:
            return named_model(model_id)

    def scaled_training_data(self, data_repo:'OzDataProvider'):
        scaled_data, scaler = self.load_data_and_scale(data_repo, start_slice = self.train_start_date, end_slice = self.train_end_date, scaler=None)
        self.trained_scaler = scaler
        return scaled_data

    def scaled_validation_data(self, data_repo:'OzDataProvider'):
        assert self.trained_scaler is not None
        scaled_data, _ = self.load_data_and_scale(data_repo, 
            start_slice = self.eval_start_date, 
            end_slice = self.eval_end_date,
            scaler = self.trained_scaler)
        return scaled_data

    def scaled_whole_data(self, data_repo:'OzDataProvider'):
        assert self.trained_scaler is not None
        scaled_data, _ = self.load_data_and_scale(data_repo, 
            start_slice = self.train_start_date, 
            end_slice = self.eval_end_date,
            scaler = self.trained_scaler)
        return scaled_data
    

    def _get_runoff_scaling_opt(self) -> Optional["TransformerMixin"]:
        if self.runoff_scaling_option == STD_SCALER:
            return StandardScaler()
        elif self.runoff_scaling_option == MINMAX_SCALER:
            return MinMaxScaler(feature_range=(-1.0, 1.0))
        else:
            return None

    def load_data_and_scale(self, d: "OzDataProvider", start_slice:datetime, end_slice:datetime, scaler:Optional['DataTransformParameters'] = None, skipna_tmax=True):
        # import xarray as xr
        # HACK:
        # The reason why I subset the data is that some of the rainfall or pet inputs at the end of the series, for some but not all stations. Ouch
        # This creates havoc with scaling, silently so, and we end up with NaNs everywhere (not that it stops TF from training, oddly.)
        # Following slicing is a stop-gap

        station_id = self.station_id

        ds: xr.Dataset = d.data_for_station(station_id, as_data_provider=False)
        rain_daily = ds.daily_series.sel(series_id=RAIN_D_COL)
        evap_daily = ds.daily_series.sel(series_id=EVAP_D_COL)
        runoff_daily = ds.daily_series.sel(series_id=RUNOFF_D_COL)
        tmax_daily = ds.daily_series.sel(series_id=TEMPMAX_D_COL)

        obs_ro: pd.Series = runoff_daily.to_series()
        date_first_obs_runoff = obs_ro[np.isfinite(obs_ro)].index[0]
        self.train_start_date = max(self.train_start_date, date_first_obs_runoff)

        # Incomplete months at the edges of the data will give incorrect monthly streamflow values. 
        from pandas.tseries.offsets import MonthEnd, MonthBegin
        start_slice = start_slice + MonthBegin(1)
        end_slice = end_slice + pd.DateOffset(months=-1) + MonthEnd(1)

        ss = slice(start_slice, end_slice)

        # rain = d.monthly_data(station_id, RAIN_KEY, cf_time=True)
        # pet = d.monthly_data(station_id, EVAP_KEY, cf_time=True)
        # runoff = d.monthly_data(station_id, RUNOFF_KEY, cf_time=True)
        # tmax = d.monthly_data(station_id, TMAX_MEAN_KEY, cf_time=True)

        rain_daily = rain_daily.sel(time=ss)
        evap_daily = evap_daily.sel(time=ss)
        runoff_daily = runoff_daily.sel(time=ss)
        tmax_daily = tmax_daily.sel(time=ss)

        rain_surplus_d = rain_daily - evap_daily
        eff_rain_d = np.maximum(0.0, rain_surplus_d)

        MONTHLY_AGG_METHOD="M" # end of month
        rsmp = {TIME_DIM_NAME: MONTHLY_AGG_METHOD}
        # we really must not do any gap filling: skipna=False

        # Caution: skipna needs to be in the stat, not the call to resample
        # https://jira.csiro.au/projects/HYDROML/issues/HYDROML-9
        # https://stackoverflow.com/questions/54461557/xarray-resampling-with-certain-nan-treatment
        rain: xr.DataArray = rain_daily.resample(indexer=rsmp).sum(skipna=False)
        pet: xr.DataArray = evap_daily.resample(indexer=rsmp).sum(skipna=False)
        tmax: xr.DataArray = tmax_daily.resample(indexer=rsmp).mean(skipna=skipna_tmax)
        rain_surplus: xr.DataArray = rain_surplus_d.resample(indexer=rsmp).sum(skipna=False)
        eff_rain: xr.DataArray = eff_rain_d.resample(indexer=rsmp).sum(skipna=False)
        runoff: xr.DataArray = runoff_daily.resample(indexer=rsmp).sum(skipna=False)

        if np.any(np.isnan(rain.values)):
            raise ValueError("monthly rainfall input data used has missing values")
        if np.any(np.isnan(pet.values)):
            raise ValueError("monthly pet input data used has missing values")
        if np.any(np.isnan(tmax.values)): # should not be possible anymore, but as a fallback if future changes... 
            raise ValueError("monthly tmax input data used has missing values")


        if scaler is None:
            scaler = DataTransformParameters(pet, rain, tmax, eff_rain, rain_surplus, runoff, self._get_runoff_scaling_opt())

        (pet_scaled, 
        rain_scaled, 
        tmax_scaled, 
        eff_rain_scaled, 
        rain_surplus_scaled, 
        runoff_scaled) = scaler.scale(pet, rain, tmax, eff_rain, rain_surplus, runoff)

        x = pd.DataFrame(
            {
                FEATURE_ID_TIME: rain.time.values,
                FEATURE_ID_RAIN: rain_scaled.values.ravel(),
                FEATURE_ID_PET: pet_scaled.values.ravel(),
                FEATURE_ID_TMAX: tmax_scaled.values.ravel(),
                FEATURE_ID_EFF_RAIN: eff_rain_scaled.values.ravel(),
                FEATURE_ID_RAIN_SURPLUS: rain_surplus_scaled.values.ravel(),
            }
        )

        if self.feature_ids is not None:
            input_features = [FEATURE_ID_TIME] + self.feature_ids
        else:
            input_features = [FEATURE_ID_TIME] + [FEATURE_ID_RAIN,
                FEATURE_ID_PET,
                FEATURE_ID_TMAX,
                FEATURE_ID_EFF_RAIN,
                FEATURE_ID_RAIN_SURPLUS,
            ][:self.num_features] # backward compatibility

        x = x[input_features]

        y = pd.DataFrame(
            {FEATURE_ID_TIME: rain.time.values, FEATURE_ID_RUNOFF: runoff_scaled.values.ravel()}
        )

        x.set_index(FEATURE_ID_TIME, inplace=True)
        y.set_index(FEATURE_ID_TIME, inplace=True)

        # Keep monthly series in untransformed space for testing/QA from notebooks
        # self.x_backtransformed = pd.DataFrame(
        #     {
        #         FEATURE_ID_TIME: rain.time.values,
        #         "rain": rain.values.ravel(),
        #         "pet": pet.values.ravel(),
        #         "tmax": tmax.values.ravel(),
        #         "eff_rain": eff_rain.values.ravel(),
        #         "rain_surplus": rain_surplus.values.ravel(),
        #     }
        # )
        # self.y_backtransformed = pd.DataFrame(
        #     {FEATURE_ID_TIME: rain.time.values, FEATURE_ID_RUNOFF: runoff.values.ravel()}
        # )
        # self.x_backtransformed.set_index(FEATURE_ID_TIME, inplace=True)
        # self.y_backtransformed.set_index(FEATURE_ID_TIME, inplace=True)

        data = Data(
            x, y, rain.to_series(), pet.to_series(), runoff.to_series(), scaler.runoff_scaler
        )
        # if stats_file is None:
        #     stats_file = 'data/ozdata/sites.csv'
        # data.stats = pd.read_csv(stats_file, header=15, index_col='siteid')
        data.source = d
        data.station_id = station_id

        return (data, scaler)

    def create_model(
        self, reload_model=False, batch_size: int = None, model_func: Callable = None, compile_model:bool=True
    ):

        if model_func is None:
            model_func = self.model_func
        if isinstance(model_func, str):
            model_func = named_model(model_func)

        if batch_size is None:
            batch_size = self.batch_size
        model = model_func(
            lstm_dim=self.lstm_dim,
            input_size=(self.seq_length, self.num_features),
            batch_shape=(batch_size, self.seq_length, self.num_features),
            stateful=self.stateful,
            seed_init=self.seed_init,
            compile_model=compile_model
            )
        if reload_model:
            assert self.model_file is not None
            _ = model.load_weights(self.model_file)
        
        return model


def mk_model_filename(model_dir, station_id, with_timestamp=True):
    if with_timestamp:
        tt = pd.Timestamp.now()
        tt_str = "{}-{:02d}-{:02d}-{:02d}-{:02d}".format(
            tt.year, tt.month, tt.day, tt.hour, tt.minute
        )
        return model_dir / ("{}_{}.hdf5".format(station_id, tt_str))
    else:
        return model_dir / ("{}.hdf5".format(station_id))


def checked_mkdir(d: Path):
    if not d.exists(): 
        d.mkdir(parents=True, exist_ok=True) # exist_ok=True here for concurrency cases
    return d


class CatchmentTraining:
    def __init__(
        self, out_dir: Optional[Path], station_id: str, data_repo: OzDataProvider, model_file: Optional[Path] = None
    ) -> None:

        self.out_dir = out_dir
        if self.out_dir is not None:
            self._pred_out_dir = checked_mkdir(self.out_dir / OUT_PREDS_DIRNAME)
            self._metrics_out_dir = checked_mkdir(self.out_dir / OUT_METRICS_DIRNAME)
            model_dir = checked_mkdir(self.out_dir / OUT_MODELS_DIRNAME)
            if model_file is None:
                model_file = mk_model_filename(model_dir, station_id)
            self.model_file = model_file
            self.plot_out_dir=self.out_dir / OUT_PLOTS_DIRNAME
        else:
            self._pred_out_dir = None
            self._metrics_out_dir = None
            self.model_file = None
            self.plot_out_dir = None

        self.conf = CatchmentTrainingConfig(station_id, model_file)
        self.conf.n_epochs = 60
        self.conf.seq_length = 24
        self.conf.steps_per_epoch = 800
        self.conf.lstm_dim = 20
        self.conf.shuffle = True
        self.conf.stride = 1
        self.conf.batch_size = 32
        self.conf.stateful = False
        # conf.train_start_date = data.flow[np.isfinite(data.flow)].index[0]
        # conf.eval_end_date = data.flow[np.isfinite(data.flow)].index[-1]
        self.data_repo = data_repo
        self.no_clobber = False

        self.fit_verbosity = (
            "auto"  # https://www.tensorflow.org/api_docs/python/tf/keras/Model
        )
        self.eval_verbosity = (
            0  # https://www.tensorflow.org/api_docs/python/tf/keras/Model
        )

        self.save_verif_report = True

        self._train_data:Optional[Data] = None
        self._eval_data:Optional[Data] = None
        self._whole_data:Optional[Data] = None

        
    @property
    def train_data(self):
        if self._train_data is None:
            self._train_data = self.scaled_training_data()
        return self._train_data

    @property
    def eval_data(self):
        if self._eval_data is None:
            self._eval_data = self.scaled_validation_data()
        return self._eval_data

    @property
    def whole_data(self):
        if self._whole_data is None:
            self._whole_data = self.scaled_whole_data()
        return self._whole_data

    def scaled_training_data(self):
        return self.conf.scaled_training_data(self.data_repo)

    def scaled_validation_data(self):
        return self.conf.scaled_validation_data(self.data_repo)

    def scaled_whole_data(self):
        return self.conf.scaled_whole_data(self.data_repo)
    
    def get_training_datasets(self, random_seed: Optional[int] = None, use_validation=False):
        train_gen, valid_gen = get_training_datasets(self.conf,
            self.train_data,
            random_seed,
            use_validation)
        return train_gen, valid_gen

    def train(self, random_seed: Optional[int] = None):
        if self.model_file is not None:
            if self.model_file.exists() and self.no_clobber:
                raise ValueError("model file {} already exists".format(self.model_file))

        use_wandb=False
        use_tensorboard=False
        if self.conf.logging != "false":
            if self.conf.logging == "wandb":
                use_wandb=True
            elif self.conf.logging == "tensorboard":
                use_tensorboard=True
                if self.conf.log_dir is None or self.conf.log_dir == "":
                    raise ValueError("use_tensorboard is True, but the log directory is not specified")
            else:
                raise ValueError(f"Unknown training logging option: {self.conf.logging}")

        self.training_result = train_model(
            self.conf,
            self.train_data,
            random_seed=random_seed,
            fit_verbosity=self.fit_verbosity,
            use_wandb=use_wandb,
            use_tensorboard=use_tensorboard,
            logdir=self.conf.log_dir,
            use_validation = self.conf.use_validation,
            es_patience=self.conf.early_stopping_patience,
            lr_factor=self.conf.lr_factor,
            lr_patience=self.conf.lr_patience,
        )
        # self.data.train_transformed_nse = self.training_result.history["nse"][-1]

        if self.model_file is not None:
            self.save_trained_weights(self.model_file)

        obs, pred, eval_metrics = eval_model(
            self.conf,
            self.train_data,
            self.conf.train_start_date,
            self.conf.train_end_date,
            eval_verbosity=self.eval_verbosity,
        )
        self.whole_data.train_nse = eval_metrics[0]
        self.whole_data.train_bias = eval_metrics[1]
        self.whole_data.obs_train = obs
        self.whole_data.pred_train = pred


    def evaluate(self):
        assert self.eval_data is not None
        assert self.whole_data is not None
        obs, pred, eval_metrics = eval_model(self.conf, self.eval_data)
        self.whole_data.eval_nse = eval_metrics[0]
        self.whole_data.eval_bias = eval_metrics[1]
        self.whole_data.obs_eval = obs
        self.whole_data.pred_eval = pred
        return obs, pred, eval_metrics

    def simulate_all_period(self):
        assert self.whole_data is not None
        obs, pred, eval_metrics = eval_model(self.conf, self.whole_data, 
            eval_start_date=self.conf.train_start_date, 
            eval_end_date=self.conf.eval_end_date)
        return obs, pred, eval_metrics

    def get_metrics(self):
        return pd.DataFrame(
            {
                "nse": [self.whole_data.train_nse, self.whole_data.eval_nse],
                "bias": [self.whole_data.train_bias, self.whole_data.eval_bias],
                "station": [self.conf.station_id, self.conf.station_id],
                "period": ["train", "evaluation"],
            }
        )

    def sync_predictions(self, save_predictions=True):
        if self._pred_out_dir is None:
            raise RuntimeError()
        assert self.eval_data is not None
        assert self._train_data is not None
        station_id = self.conf.station_id
        obs_series, pred_series, _ = self.simulate_all_period()
        predfile_all = self._pred_out_dir / "{}_series.csv".format(station_id)
        if save_predictions:
            dframe = pd.DataFrame(
                {OBS_KEY: obs_series, PRED_KEY: pred_series},
                index=obs_series.index,
            )
            dframe.to_csv(path_or_buf=predfile_all)
            metrics = self.get_metrics()
            metrics.to_csv(
                path_or_buf=(self._metrics_out_dir / "{}.csv".format(station_id))
            )
        else:
            # Read Predictions from file
            df = pd.read_csv(predfile_all, index_col=0, parse_dates=[0])
            self.eval_data.obs_eval = df.get(OBS_KEY)
            self.eval_data.pred_eval = df.get(PRED_KEY)

    def generate_report(self):
        assert self.whole_data is not None
        generate_report(
            self.conf.station_id,
            self.conf,
            self.whole_data,
            do_save=self.save_verif_report,
            out_dir=self.plot_out_dir,
        )

    def generate_training_report(self):
        assert self.whole_data is not None
        generate_training_report(
            self.conf.station_id,
            self.conf,
            self.whole_data,
            do_save=self.save_verif_report,
            out_dir=self.plot_out_dir,
        )

    def save_trained_weights(self, outfile=None):

        if outfile is None:
            weights_dir = self.out_dir / "weights"
            if not weights_dir.exists():
                weights_dir.mkdir(parents=True)
            outfile = weights_dir / self.model_file.name

        self.training_result.model.save_weights(outfile)


def get_train_valid_split(conf: "CatchmentTrainingConfig",
    data: "Data",
    use_validation=False,
    ):
    import math
    # training dates
    start_date = conf.train_start_date
    end_date = conf.train_end_date
    from pandas.tseries.offsets import DateOffset 
    d = start_date + DateOffset(months=int(conf.seq_length*4))

    if use_validation and (d < end_date):
        td = end_date - start_date
        months_n_span = int(math.floor(td.days / 30.42))
        train_months_n_span = int((4 * months_n_span) // 5)
        d_split = start_date + DateOffset(months=train_months_n_span)
        X_train = data.x[start_date:d_split]
        Y_train = data.y[start_date:d_split]
        X_valid = data.x[d_split:end_date]
        Y_valid = data.y[d_split:end_date]
    else:
        X_train = data.x[start_date:end_date]
        Y_train = data.y[start_date:end_date]
        X_valid = None
        Y_valid = None

    return X_train, Y_train, X_valid, Y_valid

def get_training_datasets(conf: "CatchmentTrainingConfig",
    data: "Data",
    random_seed: Optional[int] = None,
    use_validation=False,
):
    X_train, Y_train, X_valid, Y_valid = get_train_valid_split(conf, data, use_validation)
    train_valid = (X_valid is not None)

    _x_samples = X_train.values
    _y_samples = Y_train.values
    _batch_size = conf.batch_size
    _num_samples = conf.seq_length
    # _num_features = conf.num_features
    _stride = conf.stride
    _shuffle = conf.shuffle
    _random_seed = random_seed
    # Expected shapes of the samples by the batch generators:
    _batch_shape=(_batch_size, _num_samples, _x_samples.shape[1])
    _obs_shape=(_batch_size, 1, 1)
    output_signature=(
        tf.TensorSpec(shape=_batch_shape, dtype=tf.float32),
        tf.TensorSpec(shape=_obs_shape, dtype=tf.float32))

    # Workaround, but perhaps overall better anyway, for https://github.com/tensorflow/tensorflow/issues/58258
    train_gen = tf.data.Dataset.from_generator(
        # callable_generator,
        train_generator_batch,
        args = [_x_samples,
                _y_samples,
                _batch_size,
                _num_samples,
                _stride,
                _shuffle,
                _random_seed],
        output_signature=output_signature)


    if train_valid:
        _x_samples = X_valid.values
        _y_samples = Y_valid.values
        _batch_size = conf.batch_size
        _num_samples = conf.seq_length
        # _num_features = conf.num_features
        _stride = conf.stride
        _shuffle = conf.shuffle
        _random_seed = (random_seed + 1234) if random_seed is not None else 1234

        # Workaround, but perhaps overall better anyway, for https://github.com/tensorflow/tensorflow/issues/58258
        valid_gen = tf.data.Dataset.from_generator(
            # callable_generator,
            train_generator_batch,
                args = [_x_samples,
                        _y_samples,
                        _batch_size,
                        _num_samples,
                        _stride,
                        _shuffle,
                        _random_seed],
            output_signature=output_signature)
    else:
        valid_gen = None

    return train_gen, valid_gen


def train_model(conf: "CatchmentTrainingConfig",
    data: "Data",
    random_seed: Optional[int] = None,
    fit_verbosity="auto",
    use_wandb=False,
    use_tensorboard=False,
    logdir: Optional[Path] = None,
    use_validation=False,
    es_patience=EARLY_STOPPING_PATIENCE,
    lr_factor=REDUCE_LR_FACTOR,
    lr_patience=REDUCE_LR_PATIENCE,
):
    from keras.callbacks import (
        ModelCheckpoint,
        ReduceLROnPlateau,
        EarlyStopping,
    )

    train_gen, valid_gen = get_training_datasets(conf,
        data,
        random_seed,
        use_validation)

    if use_wandb:
        import wandb
        wandb.init(
            project="rrml",
            name=data.station_id,
            config={
                "seq_length": conf.seq_length,
                "batch_size": conf.batch_size,
                "n_epochs": conf.n_epochs,
                "steps_per_epoch": conf.steps_per_epoch,
                "shuffle": conf.shuffle,
                "stride": conf.stride,
                "model": str(conf.model_func),
                "lstm_dim": conf.lstm_dim,
                "early_stopping_patience": conf.early_stopping_patience,
                "lr_factor": conf.lr_factor,
                "lr_patience": conf.lr_patience,
                "lr_start": conf.lr_start,
                "early_stopping": conf.early_stopping,
                "dropout": conf.dropout,
                "recurrent_dropout": conf.recurrent_dropout,
                "feature_ids": conf.feature_ids,
                "num_features": conf.num_features,
            },
            reinit=True,
        )

    model = conf.create_model()

    model_checkpoint = ModelCheckpoint(
        monitor="loss", verbose=0, save_best_only=True, filepath=conf.model_file
    )

    stopping_monitor = "val_loss" if use_validation else "loss"
    callbacks = []
    # model_checkpoint,
    if conf.early_stopping:
        callbacks.append(EarlyStopping(monitor=stopping_monitor, patience=es_patience))
    callbacks.append(ReduceLROnPlateau(monitor="loss", factor=lr_factor, patience=lr_patience))

    if use_wandb:
        from wandb.keras import WandbCallback
        callbacks.append(WandbCallback())

    if use_tensorboard:
        if logdir is None or logdir == "":
            raise ValueError("use_tensorboard is True, but the log directory is not specified")
        from keras.callbacks import TensorBoard
        if isinstance(logdir, str):
            logdir = Path(logdir)
        logdir.mkdir(parents=True, exist_ok=True)
        tb = TensorBoard(logdir)
        callbacks.append(tb)

    result = model.fit(
        train_gen,
        steps_per_epoch=conf.steps_per_epoch,  # len(X) - seq_length,
        epochs=conf.n_epochs,
        callbacks=callbacks,
        validation_data=valid_gen,
        validation_steps=conf.validation_steps,
        verbose=fit_verbosity,
    )
    if use_wandb:
        wandb.finish()

    return result


def eval_generator(
    data: Data, num_samples=360, num_features=2
) -> Generator[np.ndarray, None, None]:
    while len(data) >= num_samples:
        # load X - test data
        X = [None] * num_features
        for i in range(0, num_features):
            X[i] = data.values[0:num_samples, i]

        X = np.asarray(X)
        X = X.T

        # remove used samples
        data = data[1:]

        # reshape
        X: np.ndarray = np.reshape(X, (1, num_samples, num_features))

        yield X


def np_mask_missing_values(y_true, y_pred):
    # Note there is the module np.ma, but the usage is not clear
    return remove_items_missing_observations(y_true, y_pred)

def calc_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    obs, sim = np_mask_missing_values(obs, sim)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator

    return nse_val


def relative_bias(obs: np.ndarray, sim: np.ndarray) -> float:
    """Calculate the relative bias metric.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: relative bias value: fraction of departure from total observed.
    """
    obs, sim = np_mask_missing_values(obs, sim)
    tot_obs = np.sum(obs)
    assert tot_obs > 0
    return (np.sum(sim) - tot_obs) / tot_obs


def eval_model(
    conf: "CatchmentTrainingConfig",
    data: "Data",
    eval_start_date=None,
    eval_end_date=None,
    eval_verbosity=1,
):

    if eval_start_date is None:
        eval_start_date = conf.eval_start_date
    if eval_end_date is None:
        eval_end_date = conf.eval_end_date

    X = data.x[eval_start_date:eval_end_date]
    obs = data.runoff[eval_start_date:eval_end_date]
    # obs = obs.interpolate() # WARNING: where does this come from?? No way.

    model = conf.create_model(reload_model=True, batch_size=1)
    eval_gen = eval_generator(
        X, num_samples=conf.seq_length, num_features=conf.num_features
    )

    # rescale the outputs
    preds_raw = model.predict(
        eval_gen, verbose=eval_verbosity, steps=len(X) - (conf.seq_length - 1)
    )
    preds = np.reshape(preds_raw, (-1, 1))
    preds = data.runoff_scaler.inverse_transform(preds)

    # Tensorflow now returns float32 where obs are doubles. So issue in calculations. Make uniform.
    pred = pd.Series(np.ndarray.flatten(preds).astype(np.float32), index=obs.index[-len(preds) :])
    obs_subset = obs[conf.seq_length - 1 :]
    obs_subset = pd.Series(obs_subset.values.astype(np.float32), index=obs_subset.index)

    nse = calc_nse(obs_subset, pred)
    rbias = relative_bias(obs_subset, pred)
    # print("NSE of predictions is %s" % nse)
    return obs, pred, (nse, rbias)


def is_suspicious(data, siteid):

    siteinfo = data.stats.loc[siteid]

    sus = ""
    if str(siteinfo["suspicious"]) == "nan":
        sus = True
    else:
        sus = siteinfo["suspicious"]

    return sus


def generate_report(
    station_id,
    conf: "CatchmentTrainingConfig",
    data: "Data",
    do_save=True,
    out_dir=None,
    interactive=False,
):
    """ """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(constrained_layout=True, figsize=(8.5, 11))
    gs = GridSpec(12, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0:1, :])  # rain
    ax2 = fig.add_subplot(gs[1:2, :])
    ax3 = fig.add_subplot(gs[2:3, :])
    ax1.plot(
        data.rain[conf.train_start_date : conf.train_end_date],
        color=mcolors.CSS4_COLORS["dimgrey"],
    )
    ax1.set_ylabel(MM_P_MTH)
    ax1.legend(["Rainfall"])

    ax2.plot(
        data.pet[conf.train_start_date : conf.train_end_date],
        color=mcolors.CSS4_COLORS["rosybrown"],
    )
    ax2.set_ylabel(MM_P_MTH)
    ax2.legend(["PET"])

    ax3.plot(
        data.runoff[conf.train_start_date : conf.train_end_date],
        color=mcolors.CSS4_COLORS["darksalmon"],
    )
    ax3.set_ylabel(MM_P_MTH)
    ax3.legend(["Runoff"])

    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax4 = fig.add_subplot(gs[3, 0])
    ax5 = fig.add_subplot(gs[3, 1])
    ax6 = fig.add_subplot(gs[3, 2])

    ax4.hist(
        data.rain[conf.train_start_date : conf.train_end_date],
        bins=20,
        color=mcolors.CSS4_COLORS["dimgrey"],
    )
    ax4.legend(["Rain dist"])
    ax4.set_xlabel(MM_P_MTH)
    ax4.set_ylabel("Freq")

    ax5.hist(
        data.pet[conf.train_start_date : conf.train_end_date],
        bins=20,
        color=mcolors.CSS4_COLORS["rosybrown"],
    )
    ax5.legend(["PET dist"])
    ax5.set_xlabel(MM_P_MTH)
    ax5.set_ylabel("Freq")

    ax6.hist(
        data.runoff[conf.train_start_date : conf.train_end_date],
        bins=20,
        color=mcolors.CSS4_COLORS["darksalmon"],
    )
    ax6.legend(["Runoff dist"])
    ax6.set_xlabel(MM_P_MTH)
    ax6.set_ylabel("Freq")

    ax7 = fig.add_subplot(gs[4, 0])
    ax8 = fig.add_subplot(gs[4, 1])
    ax9 = fig.add_subplot(gs[4, 2])

    ax7.hist(
        data.x["rain"][conf.train_start_date : conf.train_end_date],
        bins=20,
        color=mcolors.CSS4_COLORS["dimgrey"],
    )
    ax7.legend(["scaled rain dist"])
    ax7.set_ylabel("Freq")

    ax8.hist(
        data.x["pet"][conf.train_start_date : conf.train_end_date],
        bins=20,
        color=mcolors.CSS4_COLORS["rosybrown"],
    )
    ax8.legend(["scaled evap dist"])
    ax8.set_ylabel("Freq")

    ax9.hist(
        data.y[conf.train_start_date : conf.train_end_date],
        bins=20,
        color=mcolors.CSS4_COLORS["darksalmon"],
    )
    ax9.legend(["Scaled runoff"])
    ax9.set_ylabel("Freq")

    ax10 = fig.add_subplot(gs[5:6, :])
    ax11 = fig.add_subplot(gs[6:7, :])
    rain_ax = fig.add_subplot(gs[7:8, :])
    ax12 = fig.add_subplot(gs[8:10, 0:1])
    ax13 = fig.add_subplot(gs[8:11, 1:])
    ax14 = fig.add_subplot(gs[10, 0:1])

    # table = ax14.table(
    # cellText=[[str(data.train_nse), str(data.eval_nse)]],
    # rowLabels=[
    # "nse",
    # ],
    # colLabels=[TRAINING_LABEL, "Evaluation"],
    # loc="top",
    # )
    # table.set_fontsize(50)
    # ax14.set_xticks([])
    # ax14.set_yticks([])
    # ax14.axis("off")

    def _fn(v: float):
        return np.nan if v is None else v

    ax14.annotate(
        "Train. nse:%2.2f bias:%2.2f" % (_fn(data.train_nse), _fn(data.train_bias)),
        (0.1, 1.0),
    )
    ax14.annotate(
        "Valid. nse:%2.2f bias:%2.2f" % (_fn(data.eval_nse), _fn(data.eval_bias)),
        (0.1, 0.1),
    )
    ax14.set_xticks([])
    ax14.set_yticks([])
    ax14.axis("off")

    fig.suptitle(f"Report for {station_id}, Model-{conf.model_func.__name__}")

    ax10.plot(data.obs_eval, label="Observation", color="orange")
    ax10.set_ylabel(MM_P_MTH)
    ax10.legend(["Obs"])

    ax11.plot(data.pred_eval, label="Prediction")
    ax11.set_ylabel(MM_P_MTH)
    ax11.legend("Pred")
    ax11.set_ylim(ax10.get_ylim())
    ax11.set_xlim(ax10.get_xlim())
    rain_ax.plot(
        data.rain[conf.eval_start_date : conf.eval_end_date],
        label="Rain",
        color=mcolors.CSS4_COLORS["dimgrey"],
    )
    rain_ax.set_ylabel(MM_P_MTH)
    rain_ax.legend("Rain")

    ref_obs = data.obs_eval[-len(data.pred_eval) :]
    # TODO maybe an option to gap fill, but this may be very misleading.
    m = MaskTimeSeries(ref_obs)
    pred_masked:pd.Series = m.mask(data.pred_eval).to_series()

    v_max = max(np.nanmax(ref_obs.values), np.nanmax(pred_masked.values))
    x_range = range(0, int(v_max))

    ax12.plot(data.obs_eval[-len(data.pred_eval) :], data.pred_eval, "r+", linewidth=0)
    y_range = x_range
    ax12.plot(x_range, y_range, "b")
    ax12.set_title("Obs v Pred")
    ax12.set_xlabel("Obs")
    ax12.set_ylabel("Pred")
    ax12.legend(["Obs v Pred", "1:1"])

    ax13.plot(pred_masked.cumsum(skipna=True) * MM_TO_METRE)
    ax13.plot(ref_obs.cumsum(skipna=True) * MM_TO_METRE, color="orange")
    ax13.set_ylabel("metres")
    ax13.set_title("Cumulated runoff")

    ax13.legend(["Pred", "Obs"])

    if do_save:
        if out_dir is None:
            raise ValueError("out_fir is None") # out_dir = Path("results/plots")
        _ = checked_mkdir(out_dir)
        plotname = out_dir / "{}.png".format(station_id)
        plt.savefig(plotname)
    plt.show(block=interactive)


def generate_training_report(
    station_id,
    conf: "CatchmentTrainingConfig",
    data: "Data",
    do_save=True,
    out_dir=None,
    interactive=False,
):
    """ """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(constrained_layout=True, figsize=(8.5, 11))
    gs = GridSpec(12, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0:1, :])  # rain
    ax2 = fig.add_subplot(gs[1:2, :])
    ax3 = fig.add_subplot(gs[2:3, :])
    ax1.plot(
        data.rain[conf.train_start_date : conf.train_end_date],
        color=mcolors.CSS4_COLORS["dimgrey"],
    )
    ax1.set_ylabel(MM_P_MTH)
    ax1.legend(["Rainfall"])

    ax2.plot(
        data.pet[conf.train_start_date : conf.train_end_date],
        color=mcolors.CSS4_COLORS["rosybrown"],
    )
    ax2.set_ylabel(MM_P_MTH)
    ax2.legend(["PET"])

    ax3.plot(
        data.runoff[conf.train_start_date : conf.train_end_date],
        color=mcolors.CSS4_COLORS["darksalmon"],
    )
    ax3.plot(data.pred_train, color="blue")
    ax3.set_ylabel(MM_P_MTH)
    ax3.legend(["runoff obs.", "runoff pred"])

    # ax10 = fig.add_subplot(gs[5:6, :])
    # ax11 = fig.add_subplot(gs[6:7, :])
    # rain_ax = fig.add_subplot(gs[7:8, :])
    ax12 = fig.add_subplot(gs[8:10, 0:1])
    ax13 = fig.add_subplot(gs[8:11, 1:])
    ax14 = fig.add_subplot(gs[10, 0:1])

    # table = ax14.table(
    # cellText=[[str(data.train_nse), str(data.eval_nse)]],
    # rowLabels=[
    # "nse",
    # ],
    # colLabels=[TRAINING_LABEL, "Evaluation"],
    # loc="top",
    # )
    # table.set_fontsize(50)
    # ax14.set_xticks([])
    # ax14.set_yticks([])
    # ax14.axis("off")

    def _fn(v: float):
        return np.nan if v is None else v

    ax14.annotate(
        "Train. nse:%2.2f bias:%2.2f" % (_fn(data.train_nse), _fn(data.train_bias)),
        (0.1, 1.0),
    )
    # ax14.annotate("Valid. nse:%2.2f bias:%2.2f" % (_fn(data.eval_nse), _fn(data.eval_bias)), (0.1,0.1))
    ax14.set_xticks([])
    ax14.set_yticks([])
    ax14.axis("off")

    fig.suptitle(f"Training report for {station_id}, Model-{conf.model_func.__name__}")

    ref_obs = data.obs_train[-len(data.pred_train) :]
    # TODO maybe an option to gap fill, but this may be very misleading.
    m = MaskTimeSeries(ref_obs)
    pred_masked = m.mask(data.pred_train).to_series()

    v_max = max(np.nanmax(ref_obs.values), np.nanmax(pred_masked.values))
    x_range = range(0, int(v_max))

    ax12.plot(
        data.obs_train[-len(data.pred_train) :], data.pred_train, "r+", linewidth=0
    )
    y_range = x_range
    ax12.plot(x_range, y_range, "b")
    ax12.set_title("Obs v Pred")
    ax12.set_xlabel("Obs")
    ax12.set_ylabel("Pred")
    ax12.legend(["Obs v Pred", "1:1"])

    ax13.plot(pred_masked.cumsum(skipna=True) * MM_TO_METRE)
    ax13.plot(ref_obs.cumsum(skipna=True) * MM_TO_METRE, color="orange")
    ax13.set_ylabel("metres")
    ax13.set_title("Cumulated runoff")

    ax13.legend(["Pred", "Obs"])

    if do_save:
        if out_dir is None:
            raise ValueError("out_fir is None") # out_dir = Path("results/plots")
        _ = checked_mkdir(out_dir)
        plotname = out_dir / "{}.png".format(station_id)
        plt.savefig(plotname)
    plt.show(block=interactive)


from keras.callbacks import Callback

class LRFinder(Callback):
    """Callback that exponentially adjusts the learning rate after each training batch between start_lr and
    end_lr for a maximum number of batches: max_step. The loss and learning rate are recorded at each step allowing
    visually finding a good learning rate as per [Sylvain Gugger's post](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html) via
    the plot method.

    Credit for this class: [Andrich van Wyk](https://www.avanwyk.com/finding-a-learning-rate-in-tensorflow-2/)
    """

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, max_steps: int = 100, smoothing=0.9):
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        step = self.step
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True

        if step == self.max_steps:
            self.model.stop_training = True

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def plot(self):
        from matplotlib.ticker import FormatStrFormatter
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
        # ax.set_title(loss)
        ax.plot(self.lrs, self.losses)

    def slope(self, smooth=True):
        losses = np.array(self.losses)
        slope = losses[1:] - losses[:-1]
        if smooth:
            kernel_size = 10
            kernel = np.ones(kernel_size) / kernel_size
            slope = np.convolve(slope, kernel, mode='same')
        return slope

    def slope_plot(self, smooth=True):
        slope = self.slope(smooth=smooth)
        from matplotlib.ticker import FormatStrFormatter
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss delta')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
        ax.plot(self.lrs[1:], slope)


    def guessed_lr(self):
        indx = np.argmin(self.slope())
        return self.lrs[indx+1]
