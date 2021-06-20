import matplotlib.pyplot as plt
import os
import settings


def verify_if_pretrained_model_exists(path_to_model):
    if os.path.isfile(path_to_model) and os.access(path_to_model, os.R_OK):
        return True
    else:
        return False


# Saving plot with Test Predictions
def save_plot_test(y_test, predicted_test_prices, company):
    plt.plot(y_test, color='blue', label=f'Actual {company} price')
    plt.plot(predicted_test_prices, color='red', label=f'Predicted {company} price')
    plt.title(f'{company} Price')
    plt.xlabel('Time')
    plt.ylabel(f'{company} Stock Price')
    plt.savefig(f'{settings.result_of_test_figures_path}/test_for_{company}.png')