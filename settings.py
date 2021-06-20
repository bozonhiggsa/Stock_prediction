# Companies
provider = 'yahoo'
companies = ['NFLX', 'FB', 'AMZN', 'TSLA', 'GOOG', 'AAPL']
mode = ["train", "predict"]

# Number of days for prediction each next day during training
days_for_prediction_on_train = 50

days_for_validation = 300

days_for_test = 100

# define the time as "yyyy-mm-dd"
start_date = (2012, 1, 1)

models_path = 'models'
result_of_test_figures_path = 'test_results'

# hyperparameters of the model
LSTM_units_1st_layer = 60
LSTM_units_2nd_layer = 60
dropout = 0.2
Dense_1st_layer = 64
Dense_2nd_layer = 32
epochs = 30
batch_size = 64


