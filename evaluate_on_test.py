import settings
from tensorflow.keras.models import load_model
import utils


def estimate_on_test(x_test, y_test, scaler, company):
    path_to_model = f'{settings.models_path}/best_model_{company}.h5'
    if utils.verify_if_pretrained_model_exists(path_to_model):
        # Make Predictions on Test Data
        model = load_model(path_to_model, compile=False)
        predicted_test_prices_scaled = model.predict(x_test)
        predicted_test_prices = scaler.inverse_transform(predicted_test_prices_scaled)
    else:
        print('There is no pretrained model for evaluation on test data!')
        return

    # Saving plot with Test Predictions
    utils.save_plot_test(y_test, predicted_test_prices, company)
