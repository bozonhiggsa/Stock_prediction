import datetime as dt
import pandas_datareader as web
import settings


class DataLoader:
    def __init__(self, company):
        self.company = company

    def load_data(self):
        start_time = dt.datetime(settings.start_date[0], settings.start_date[1], settings.start_date[2])
        # last day for full training - yesterday
        end_time_full = dt.datetime.now() - dt.timedelta(days=1)
        # last day for preliminary training - yesterday minus days_for_test ago
        end_time_preliminary = dt.datetime.now() - dt.timedelta(days=1) - dt.timedelta(days=settings.days_for_test)

        data_full = web.DataReader(self.company, settings.provider, start_time, end_time_full)
        data_preliminary_train = web.DataReader(self.company, settings.provider, start_time, end_time_preliminary)

        return data_full, data_preliminary_train

