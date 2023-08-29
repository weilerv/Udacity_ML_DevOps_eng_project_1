'''
Create testing and logging for churn_library.py functions
Virág Weiler
August 2023
'''

import os
import logging
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models
from os.path import exists
import numpy as np
import pytest

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    pytest.df = df


def test_eda():
    '''
    test perform eda function
    '''
    # check if all pictures were created to the right folder
    df = pytest.df
    try:
        perform_eda(df)
        logging.info('SUCCESS: perform_eda function run without error')
    except Exception as e:
        logging.error('There were issues while running perform_eda fuction')
        raise e

    try:
        assert os.path.exists('./images/eda/churn_hist.png')
        assert os.path.exists('./images/eda/customer_age_hist.png')
        assert os.path.exists('./images/eda/material_status_bar.png')
        assert os.path.exists('./images/eda/total_trans_ct_hist.png')
        assert os.path.exists('./images/eda/heatmap.png')
        logging.info('SUCCESS: all files were found')
    except AssertionError as err:
        logging.error(
            'Not all files were found where they should have been created.')
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    df = pytest.df
    df = encoder_helper(df, [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ])
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info('SUCCESS: df was encoded and shape is not null')
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    pytest.df = df


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    df = pytest.df
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    # test X_train
    try:
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        logging.info('SUCCESS: X_train df has rows and columns > 0')
    except AssertionError as err:
        logging.error('X_train data appears not to have nows or columns')
    # test_X_test
    try:
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        logging.info('SUCCESS: X_test df has rows and columns > 0')
    except AssertionError as err:
        logging.error('X_test data appears not to have nows or columns')
    # test y_train
    try:
        assert y_train.shape[0] > 0
        logging.info('SUCCESS: y_train df has rows and columns > 0')
    except AssertionError as err:
        logging.error('y_train data appears not to have nows or columns')
    # test y_test
    try:
        assert y_test.shape[0] > 0
        logging.info('SUCCESS: y_test df has rows and columns > 0')
    except AssertionError as err:
        logging.error('y_test data appears not to have nows or columns')
    pytest.X_train = X_train
    pytest.X_test = X_test
    pytest.y_train = y_train
    pytest.y_test = y_test


def test_train_models():
    '''
    test train_models
    '''
    X_train = pytest.X_train
    X_test = pytest.X_test
    y_train = pytest.y_train
    y_test = pytest.y_test
    train_models(X_train, X_test, y_train, y_test)
    # check if all plots, pictures were created
    try:
        assert os.path.exists('./images/results/random_forest_scores.png')
        assert os.path.exists(
            './images/results/logistic_regression_scores.png')
        assert os.path.exists('./images/results/logistic_regression_roc.png')
        assert os.path.exists('./images/results/random_forest_roc.png')
        assert os.path.exists('./images/results/roc_both_models.png')
        logging.info('SUCCESS: all files were found')
    except AssertionError as err:
        logging.error(
            'Not all files were found where they should have been created.')
        raise err

    # check if both models were saved
    try:
        assert os.path.exists('./models/rfc_model.pkl')
        assert os.path.exists('./models/logistic_model.pkl')
        logging.info('SUCCESS: all models were saved')
    except AssertionError as err:
        logging.error('Not all files were saved')
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
