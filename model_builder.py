import settings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint


def model_build_and_train(x_train, y_train, company):
    model = Sequential()

    model.add(LSTM(units=settings.LSTM_units_1st_layer, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(settings.dropout))
    model.add(LSTM(units=settings.LSTM_units_2nd_layer))
    model.add(Dropout(settings.dropout))
    model.add(Dense(units=settings.Dense_1st_layer))
    model.add(Dense(units=settings.Dense_2nd_layer))
    model.add(Dense(units=1))

    # saving the best model
    сheckpoint = ModelCheckpoint(f'models/best_model_{company}.h5', monitor='val_loss', save_best_only=True, verbose=1)

    # Train the Model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train[:(len(x_train) - settings.days_for_validation)],
              y_train[:(len(x_train) - settings.days_for_validation)],
              epochs=settings.epochs, batch_size=settings.batch_size,
              validation_data=(x_train[(len(x_train) - settings.days_for_validation):],
              y_train[(len(x_train) - settings.days_for_validation):]), shuffle=False, callbacks=[сheckpoint])

