import numpy as np
import settings


def prepare_data_for_training(data_preliminary_train, scaled_data):
    days_for_prediction_on_train = settings.days_for_prediction_on_train

    # Prepare the Training Dataset
    x_train = []
    y_train = []

    # We use days_for_prediction_on_train (days) to predict 1 next day
    for x in range(days_for_prediction_on_train, len(scaled_data)):
        x_train.append(scaled_data[x - days_for_prediction_on_train: x, 0])
        y_train.append(scaled_data[x, 0])

    # therefor x_train is the list with ndarrays inside
    # and each ndarray consists of days_for_prediction_on_train values of price
    # we need to convert the list to ndarray
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Do reshape in order to add dimension for batch
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train


def prepare_data_for_test(data_full, scaler):
    days_for_prediction_on_train = settings.days_for_prediction_on_train
    days_for_test = settings.days_for_test

    test_data_and_part_of_train =\
        data_full['Adj Close'][len(data_full) - days_for_test -
                               days_for_prediction_on_train:].values

    # Convert the Series to a Numpy array and transform to column
    test_data_and_part_of_train = test_data_and_part_of_train.reshape(-1, 1)
    scaled_data_test = scaler.transform(test_data_and_part_of_train)

    # Prepare the Test Dataset
    x_test = []
    y_test = []

    # We use days_for_prediction_on_train (days) to predict 1 next day
    for x in range(days_for_prediction_on_train, len(scaled_data_test)):
        x_test.append(scaled_data_test[x - days_for_prediction_on_train: x, 0])
        y_test.append(test_data_and_part_of_train[x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_test, y_test


def prepare_data_for_predict(data_full, scaler):
    days_for_prediction_on_train = settings.days_for_prediction_on_train

    data_for_current_prediction =\
        data_full['Adj Close'][len(data_full) - days_for_prediction_on_train: len(data_full) + 1].values

    # Convert the Series to a Numpy array and transform to column
    data_for_current_prediction = data_for_current_prediction.reshape(-1, 1)

    data_for_current_prediction_scaled = scaler.transform(data_for_current_prediction)
    data_for_current_prediction_scaled = np.reshape(data_for_current_prediction_scaled,
                                                    (1, data_for_current_prediction_scaled.shape[0], 1))

    return data_for_current_prediction_scaled

