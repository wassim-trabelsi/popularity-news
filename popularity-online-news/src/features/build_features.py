# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    df = pd.read_csv(input_filepath)

    # Keep only the relevant columns
    important_features = ['kw_avg_avg','weekday_is_saturday', 'self_reference_avg_sharess', 'average_token_length', 
                      'n_unique_tokens', 'avg_positive_polarity', 'num_hrefs', 'global_subjectivity', 'num_videos']
    
    if 'popularity' in df.columns:
        df = df[important_features + ['popularity']]
    else:
        df = df[important_features]
    
    # Remove the outliers (only for training data)
    if 'train' in input_filepath:
        df = df[df['num_hrefs'] < 250]
    
    # Remove rows with missing values
    if 'train' in input_filepath:
        df = df.dropna()
    
    df.reset_index(drop=True, inplace=True)
    
    columns = df.columns

    # Scale the data
    if 'train' in input_filepath:
        # Remove the target variable
        target = df.pop('popularity')
        columns = df.columns
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        scaler_filename = os.path.split(output_filepath)[0] + '/scaler.joblib'
        joblib.dump(scaler, scaler_filename)
        logger.info('Scaling successfully done and saved')
        df = pd.DataFrame(df, columns=columns)
        df['popularity'] = target
    else:
        if 'popularity' in df.columns:
            replug_it = True
            target = df.pop('popularity')
        columns = df.columns
        if not os.path.exists(os.path.split(output_filepath)[0] + '/scaler.joblib'):
            logger.error('No scaler found, please train the model first')
            raise Exception('No scaler found, please train the model first')
        scaler_filename = os.path.split(output_filepath)[0] + '/scaler.joblib'
        scaler = joblib.load(scaler_filename)
        df = scaler.transform(df)
        if replug_it:
            df = pd.DataFrame(df, columns=columns)
            df['popularity'] = target
        else:
            df = pd.DataFrame(df, columns=columns)
        logger.info('Scaling successfully loaded and done')

    # Save the data
    df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
