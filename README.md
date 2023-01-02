# Probabilistic Entity Resolution

## Setup the Conda Environment
To install this package, run the following:

```bash
conda create -y --name s2and python==3.7
conda activate s2and
pip install -r requirements.in
pip install -e .
```

If you run into cryptic errors about GCC on macOS while installing the requirments, try this instead:
```bash
CFLAGS='-stdlib=libc++' pip install -r requirements.in
```

## Download S2AND Data 
To obtain the S2AND dataset, run the following command after the package is installed (from inside the `S2AND` directory):  
```[Expected download size is: 50.4 GiB]```

`aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release data/`

Note that this software package comes with tools specifically designed to access and model the dataset.

## Setup Configuration
Modify the config file at `data/path_config.json`. This file should look like this
```
{
    "main_data_dir": "absolute path to wherever you downloaded the data to",
    "internal_data_dir": "ignore this one unless you work at AI2"
}
```
As the dummy file says, `main_data_dir` should be set to the location of wherever you downloaded the data to, and
`internal_data_dir` can be ignored, as it is used for some scripts that rely on unreleased data, internal to Semantic Scholar.

## Preprocess Dataset
Run the Preprocessing step for each dataset, this step creates the following directory structure:
```
/data
 -> /{dataset}
    -> /seed{seed #}
        -> pickle files stored here
```

2 kinds of pickle files are created and stored for each split of the data (train/test/val), following 
this naming convention: train_features.pkl, train_signatures.pkl.

The features pickle contains a dictionary of type: 
```Dict[block_id: str, Tuple[features: np.ndarray, labels: np.ndarray, cluste_ids: np.ndarray]]```. 
NOTE: The pairwise features are compressed in order to be stored as a n(n-1)/2 matrix rather than an nxn symmetric matrix.
The signatures pickle contains all the metadata for each signature in a block.

Sample command:
```commandline
python pipeline/preprocess_s2and_data.py --data_home_dir="./data" --dataset_name="pubmed"
```

## End-to-end model training
The end-to-end model is defined in file pipeline/model.py. For training of this model, run python script
sample_scripts/train_e2e_model.py

Sample Command:
```commandline
python sample_scripts/train_e2e_model.py --wandb_run_params=configs/wandb_overfit_1_batch.json
```

