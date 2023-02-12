# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import pickle

@click.command()
@click.argument('train_filepath', type=click.Path(exists=True))
@click.argument('models_dir', type=click.Path())
def main(train_filepath, models_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
    data = pd.read_csv(train_filepath)
    X = data.drop(['popularity'], axis=1)
    y = data['popularity']
    rf.fit(X, y)
    y_pred = rf.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)

    logger.info(f"Model successfully trained : Train mse = {mse:.2f}, Train RMSE = {rmse:.2f}")

    # Save the model
    with open(models_dir + '/rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
