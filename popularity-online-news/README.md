popularity-online-news
==============================

On articles published by Mashable (mashable.com), let's determine weather an article is popular or not based on social linking and article features. 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    |   |   └── OnlineNewsPopularity.names       
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    |       └── OnlineNewsPopularity.csv             
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

# Get Started

## Make dataset 

1. Go to the root of the project and run the following command:

```bash
python src/data/make_dataset.py data/raw/OnlineNewsPopularity.csv data/interim
```

 This will create a new file in the interim folder called `train.csv` and `test.csv` with the train and test data respectively.

2. Then generate the features by running the following command:

```bash
python src/features/build_features.py data/interim/train.csv data/processed/train.csv
```

```bash
python src/features/build_features.py data/interim/test.csv data/processed/test.csv
```


## Train model

```bash
python src/models/train_model.py data/processed/train.csv models
```

## Predict model

```bash
python src/models/predict_model.py data/processed/test.csv models data/processed/predictions.csv
```


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
