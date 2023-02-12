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
@click.argument('test_filepath', type=click.Path(exists=True))
@click.argument('models_dir', type=click.Path(exists=True))
@click.argument('preds_filepath', type=click.Path())
def main(test_filepath, models_dir, preds_filepath):
    """ 
    Read the test data and make predictions using the trained model.
    """
    logger = logging.getLogger(__name__)
    data = pd.read_csv(test_filepath)
    if 'popularity' in data.columns:
        X = data.drop(['popularity'], axis=1)
        y = data['popularity']
    else:
        X = data
    rf = pickle.load(open(models_dir + '/rf_model.pkl', 'rb'))
    y_pred = rf.predict(X)

    if 'popularity' in data.columns:
        mse = mean_squared_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        logger.info(f"Model successfully tested : test mse = {mse:.2f}, test RMSE = {rmse:.2f}")

    # Save the predictions
    pd.DataFrame(y_pred, columns=['predicted']).to_csv(preds_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()