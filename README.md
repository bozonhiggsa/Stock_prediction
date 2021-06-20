### Stock market prediction
##### Implementation for several financial instruments - NasdaqGS Price (NFLX, FB, AMZN, TSLA, GOOG, AAPL)

Technologies:
- Python 3;
- TensorFlow 2, Keras;
- pandas;
- pandas_datareader;
- numpy;
- scikit-learn;
- matplotlib.pyplot;
- argparse.

Data source:
finance.yahoo.com

To Run:
- python3 predictor.py --mode Mode --company Company

By default:
- Mode == 'predict' (also we can use Mode == 'train');
- Company == 'NFLX' (also we can use as Company one of ['FB', 'AMZN', 'TSLA', 'GOOG', 'AAPL'] )

#### License

This project is licensed under the terms of the MIT license.