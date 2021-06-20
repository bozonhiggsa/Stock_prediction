import argparse
import os
import settings
import utils
from sklearn.preprocessing import MinMaxScaler
from data_loader import DataLoader
from tensorflow.keras.models import load_model
from data_handler import prepare_data_for_training, prepare_data_for_test, prepare_data_for_predict
from model_builder import model_build_and_train
from evaluate_on_test import estimate_on_test


if __name__ == '__main__':
    # arguments parser for running directly from the command line
    parser = argparse.ArgumentParser(description='Parameters for training and prediction')
    parser.add_argument('--mode', '-m', help='Mode - train or predict', default='predict')
    parser.add_argument('--company', '-c', help='Company Name', default='NFLX')
    args = parser.parse_args()

    # verify arguments from command line
    if args.mode not in settings.mode:
        raise Exception("Mode value is incorrect")
    if args.company not in settings.companies:
        raise Exception("Company name is incorrect")

    # create folder to store the best models, if it doesnt exist
    if not os.path.exists(settings.models_path):
        os.makedirs(settings.models_path)

    # create folder to store figures with test results, if it doesnt exist
    if not os.path.exists(settings.result_of_test_figures_path):
        os.makedirs(settings.result_of_test_figures_path)

    # Load the data
    dataloader = DataLoader(args.company)
    data_full, data_preliminary_train = dataloader.load_data()

    scaler = MinMaxScaler(feature_range=(0, 1))
    # Convert the Series to a Numpy array and transform to column
    data_train_close = data_preliminary_train['Adj Close'].values.reshape(-1, 1)
    # Scale the Train Data
    scaled_data = scaler.fit_transform(data_train_close)

    # Main processing
    if args.mode == 'train':
        x_train, y_train = prepare_data_for_training(data_preliminary_train, scaled_data)
        model_build_and_train(x_train, y_train, args.company)
        x_test, y_test = prepare_data_for_test(data_full, scaler)
        estimate_on_test(x_test, y_test, scaler, args.company)
        print('Model is ready and saved, plot with results on test data is saved')
    elif args.mode == 'predict':
        data_for_current_prediction_scaled = prepare_data_for_predict(data_full, scaler)
        path_to_model = f'{settings.models_path}/best_model_{args.company}.h5'
        if utils.verify_if_pretrained_model_exists(path_to_model):
            model = load_model(path_to_model, compile=False)
            predicted_price_today = model.predict(data_for_current_prediction_scaled)
            predicted_price_today = scaler.inverse_transform(predicted_price_today)

            print(f'Prediction ({args.company}) for now: {predicted_price_today}')
        else:
            print('There is no pretrained model for prediction for current day!')

