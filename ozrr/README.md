# OZRR

[![license](https://img.shields.io/badge/license-BSD-blue.svg)](https://github.com/csiro-hydroinformatics/monthly-lstm-runoff/blob/master/LICENSE) ![status](https://img.shields.io/badge/status-beta-orange.svg)

## Scope

A python package with modules to support modelling for a paper using deep learning and/or process based models for runoff modelling. The main functional blocks are:

* Loading data from a dataset by [Lerat et. al. (2020)](https://doi.org/10.1016/j.jhydrol.2020.125129). It is now handled in a standalone package [ozrr-data](https://github.com/csiro-hydroinformatics/ozrr-data)
* Facilities for the production of derived data from raw inputs (data Q/A and transform/scaling)
* High level constructs to streamline the batch calibration of "process" based models (PBM) and teosorflow/pytorch backed deep learning models

## Quickstart

Installation instructions are, as of 2023-04, on [this web site](https://csiro-hydroinformatics.github.io/monthly-lstm-runoff/)

With minimal adaptation you should be able to execute  [../notebooks/tf_models.py](../notebooks/tf_models.py) or its traditional jupyter notebook equivalent [../notebooks/tf_models.ipynb](../notebooks/tf_models.ipynb)