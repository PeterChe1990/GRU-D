# GRU-D
This is a re-implementation of the `GRU-D` model with `Python3 + Keras2 + Tensorflow`.


## Reference
Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu. ["Recurrent Neural Networks for Multivariate Time Series with Missing Values"](https://www.nature.com/articles/s41598-018-24271-9), Scientific Reports (SREP), 8(1):6085, 2018.

An earlier version is available on arXiv ([arXiv preprint arXiv:1606.01865](https://arxiv.org/abs/1606.01865)).


## Requirements
* `Python3` packages
	* Keras>=2.2.0
	* jupyter>=1.0.0
	* notebook>=5.4.0
	* numpy>=1.14.0
	* scikit-learn>=0.19.1
	* tensorboard>=1.10.0
	* tensorflow>=1.7.0


## Running on your own data
* We use `[WD]` to represent the working directory (i.e., working path, in which all related data/log/model/result/evaluation files are stored).
* We assume the data are saved in the folder `[WD]/data/${dataset_name}`. Please see [`data_handler.py`](data_handler.py) for more information.
	* In `[WD]/data/${dataset_name}/data.npz`, there are `input`, `masking`, `timestamp`, `label_${label_name}`. Each of them is of the shape `(n_samples, ...)`
	* In `[WD]/data/${dataset_name}/fold.npz`, there are `fold_${label_name}`, `mean_${label_name}`, `std_${label_name}`. Each of them is of the shape `(k_fold, 3, ...)`, for train/validation/test sets in k-fold cross validation.
* Our GRU models take `(x, masking, timestamp)` as the inputs. Please refer to [`models.py`](models.py) and [`nn_utils/grud_layers.py`](nn_utils/grud_layers.py).
* [`Run.ipynb`](Run.ipynb) serves as an example script for model training and evaluation.


## Running on [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/)
The following steps will help you to conduct experiments for mortality predictions on the MIMIC-III dataset with the time series within the first 48 hours after the patient's admission.
We rely on this Benchmarking [codebase](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII) to extract and preprocess the time series data from the [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/) dataset and provide necessary scripts to convert the data for our GRU-D models.

1. Make sure you have the latest benchmarking [codebase](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII) and set up the database connection in the [_Requirements_](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII/blob/master/readme.md#database) section.
2. Follow steps 1-6 in the [_Select admissions and all features_](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII/blob/master/readme.md#select-admissions-and-all-features) section. I.e., execute
	* All __11__ scripts named as `[#]_***.ipynb` for `0 <= [#] <= 9`
	* Some scripts (e.g., `8_processing.ipynb`) may take hours or a couple of days to complete.
3. Follow steps 1,2,4 in the [_Generate 17 processed features, 17 raw features and 140 raw features_](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII/blob/master/readme.md#generate-17-processed-features-17-raw-features-and-140-raw-features) section. I.e., execute
	* `run_necessary_sqls.ipynb`
	* `10_get_17-features-processed.ipynb`
	* `10_get_99plus-features-raw.ipynb`
4. Follow step 3 in the [_Generate time series_](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII/blob/master/readme.md#generate-time-series) section with X=48(hours). I.e., execute
	* `11_get_time_series_sample_99plus-features-raw_48hrs.ipynb`
5. Now you should have extracted necessary data files from the benchmarking codebase. Please set the directories in [`Prepare-MIMIC-III-data.ipynb`](Prepare-MIMIC-III-data.ipynb) and execute it to prepare the data for GRU-D.
6. Execute [`Run.ipynb`](Run.ipynb) and check the results!
