import tensorflow as tf
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

from model_nn import NNModel

FILENAME_TRAIN = "./data/train.csv"
FILENAME_TEST = "./data/test.csv"
FILENAME_SUBMIT = "./submit/submit.csv"


# Model Parameters
NN_EPOCHS = 10000


def create_column_index_dictionary(column_names):
    d = dict()
    for idx, cname in enumerate(column_names):
        d[cname] = idx
    return d

# All columns
PREDICTION_COLUMN = "SalePrice"

ALL_COLUMNS = ['Id', 'LotArea', 'MSSubClass', 'LotFrontage', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
               'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
               'HeatingQC',
               '1stFlrSF', '2ndFlrSF',
               'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
               'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
               'WoodDeckSF']

FEATURE_COLUMNS = ['LotArea', 'LotFrontage', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                   'MasVnrArea',
                   '1stFlrSF', '2ndFlrSF',
                   'LowQualFinSF', 'GrLivArea',
                   'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
                   'FullBath'
                   # 'HeatingQC'

                   ]
# Basement related features:
# 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'


# Bad feaures:
# BsmtFullBath, BsmtHalfBath

COLUMN_INDEXES = create_column_index_dictionary(FEATURE_COLUMNS)


# Regression columns - They are given as numbers whose distance means something

# LotFrontage - replace N/As with 0s
REGRESSION_COLUMNS = ['LotArea', 'LotFrontage', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                      'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                      'GrLivArea',  'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
                      'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF']


# Class column - They are given as either numbers (distance means nothing) or strings (must encode to vector)

# MSSubClass - No idea what is this
# GarageYrBlt - mostly regressive, however, it contains N/As, so we must think of a way to convert that
#               it could be potentially encoded as Vector([0, 0] for N/A, [1, YEAR]) for garage and when was it built

CLASS_COLUMNS = ['MSSubClass', 'GarageYrBlt', 'HeatingQC']


def plot(train_x, train_y, x_column_name):
    plt.plot(train_x[:, COLUMN_INDEXES[x_column_name]], train_y, 'rx')
    # plt.axis([0, 40000, 0, 500000])
    plt.xlabel(x_column_name)
    plt.ylabel(PREDICTION_COLUMN)
    plt.show()


def check_for_nans(X, data_set_name):
    for fc in FEATURE_COLUMNS:
        nan_rows = np.where(pd.isnull(X[:, COLUMN_INDEXES[fc]]))[0]
        if len(nan_rows) > 0:
            print("Found bad regression column [{:s}] = {:s}\t\t\tExample row: ({:d})".format(data_set_name, fc,
                                                                                             nan_rows[0]))


def fix_bad_regression_columns(X, data_set_name):
    def fix_nans_with_zeros(column_name):
        X[np.where(pd.isnull(X[:, COLUMN_INDEXES[column_name]]))] = 0

    fix_nans_with_zeros('LotArea')
    fix_nans_with_zeros('LotFrontage')
    fix_nans_with_zeros('MasVnrArea')
    # fix_nans_with_zeros('BsmtFinSF1')
    # fix_nans_with_zeros('BsmtFinSF2')
    # fix_nans_with_zeros('BsmtUnfSF')
    # fix_nans_with_zeros('TotalBsmtSF')
    fix_nans_with_zeros('GarageCars')
    fix_nans_with_zeros('GarageArea')
    # fix_nans_with_zeros('BsmtFullBath')
    # fix_nans_with_zeros('BsmtHalfBath')
    check_for_nans(X, data_set_name)
    return




def encode_class_column(X, col_name):
    if col_name == "HeatingQC":
        # Ex	Excellent
        # Gd	Good
        # TA	Average/Typical
        # Fa	Fair
        # Po	Poor
        values = ['Po', 'Fa', 'TA', 'Gd', 'Ex']


        for quality, str_q in enumerate(values):
            return None
            # if str_q
            # return [quality]


def encode_classification_columns(X, data_set_name):
    fixed = []

    for cn in FEATURE_COLUMNS:
        if cn in REGRESSION_COLUMNS:
            fixed.append(X[:, COLUMN_INDEXES[cn]])
        elif cn in CLASS_COLUMNS:
            print(cn)
            fixed.append(encode_class_column(X[:, COLUMN_INDEXES[cn]], cn))
        else:
            assert False

    return np.transpose(np.array(fixed))


def normalize_data(x, y, mean_x=None, std_x=None, mean_y=None, std_y=None):
    assert (mean_x is None) == (std_x is None) and (mean_y is None) == (std_y is None) and\
           (mean_x is None) == (mean_y is None)

    if mean_x is None:
        mean_x = np.mean(x, axis=0)
        std_x = np.std(x, axis=0)
        mean_y = np.mean(y, axis=0)
        std_y = np.std(y, axis=0)

    normalized_x = (x - mean_x) / std_x
    normalized_y = (y - mean_y) / std_y
    return normalized_x, normalized_y, mean_x, std_x, mean_y, std_y


def run_nn_model(train_x, train_y, test_x):
    print("\nRunning Neural Network model...")
    # Number of features for the neural network should be extracted from X, i.e. it's the number of columns in X
    nn_model = NNModel(train_x.shape[1])
    fit_losses = nn_model.fit(train_x, train_y, epochs=NN_EPOCHS, verbose=True, verbose_step=1000)
    # TODO: Analyse losses?
    print("[DONE] Running Neural Network model.\n")
    return nn_model.predict(train_x), nn_model.predict(test_x)


def merge_other_predictions(data_x, nn_predictions):
    if nn_predictions is not None:
        x = np.concatenate((data_x, nn_predictions), axis=1)
        return x
    else:
        return data_x


def run_xgboost_model(train_x, train_y, test_x, nn_predictions):
    def split(predictions):
        if predictions is None:
            return None, None
        else:
            train_p, test_p = predictions
            return train_p, test_p

    print("Running xgboost model...")
    regressor = XGBRegressor(learning_rate=0.5)  # TODO: Provide parameters

    # TODO: Merge with other predictions
    train_nn_predictions, test_nn_predictions = split(nn_predictions)
    X = merge_other_predictions(train_x, train_nn_predictions)
    y = train_y

    regressor.fit(X, y, verbose=True)

    # Prediction part
    t_X = merge_other_predictions(test_x, test_nn_predictions)
    print("[DONE] Running xgboost model.")
    return regressor.predict(t_X)


def run():
    train_df = pd.read_csv(FILENAME_TRAIN, header=0)
    test_df = pd.read_csv(FILENAME_TEST, header=0)

    train_x = train_df[FEATURE_COLUMNS].values
    train_y = train_df[[PREDICTION_COLUMN]].values

    test_x = test_df[FEATURE_COLUMNS].values

    fix_bad_regression_columns(train_x, data_set_name="TRAIN")
    fix_bad_regression_columns(test_x, data_set_name="TEST")

    train_x = encode_classification_columns(train_x, data_set_name="TRAIN")
    test_x = encode_classification_columns(test_x, data_set_name="TEST")

    train_x, train_y, mean_x, std_x, mean_y, std_y = normalize_data(train_x, train_y)
    test_x = (test_x - mean_x) / std_x

    # TODO: Change test_x=train_x once test data is read
    nn_predictions_train, nn_predictions_test = run_nn_model(train_x, train_y, test_x=test_x)
    # nn_predictions_train, nn_predictions_test = None, None

    # TODO: To pass in other different models
    xgboost_predictions = run_xgboost_model(train_x, train_y, test_x=test_x,
                                            nn_predictions=(nn_predictions_train, nn_predictions_test))

    # Denormalize and reshape into column vector
    final_predictions = xgboost_predictions * std_y + mean_y
    final_predictions = final_predictions.reshape(len(final_predictions), 1)

    # Merge with IDs
    test_ids = test_df[['Id']].values
    final_data = np.concatenate((test_ids, final_predictions), axis=1).astype(dtype=np.int32)

    # Save data
    pd.DataFrame(final_data, columns=['Id', 'SalePrice']).to_csv(FILENAME_SUBMIT, index=False)


    # plot(train_x, train_y, 'YrSold')



run()



