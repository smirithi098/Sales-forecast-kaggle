#%% import libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import holidays

#%% Import data

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data.date = pd.to_datetime(train_data.date)
test_data['date'] = pd.to_datetime(test_data['date'])

#%% Create columns for day of week

def get_week_day(d):
    return int(datetime.strftime(d, "%w"))+1

train_data['week_day'] = train_data['date'].apply(get_week_day)
test_data['week_day'] = test_data['date'].apply(get_week_day)


#%% extract the year and month values from date

train_data['year'] = train_data['date'].apply(lambda d: int(datetime.strftime(d, "%Y")))
train_data['month'] = train_data['date'].apply(lambda d: int(datetime.strftime(d, "%m")))

test_data['year'] = test_data['date'].apply(lambda d: int(datetime.strftime(d, "%Y")))
test_data['month'] = test_data['date'].apply(lambda d: int(datetime.strftime(d, "%m")))

#%% get the list of holidays for all countries in all years

years = np.concatenate((train_data['year'].unique(), test_data['year'].unique()), axis=None).tolist()
countries = train_data['country'].unique().tolist()

def get_holiday_list(country, year):
    list_name = f'{country}_holidays'
    holiday_list = holidays.country_holidays(country, years=year, language='en_US')
    train_data[list_name] = train_data.loc[train_data['country'] == country, 'date'].apply(
        lambda x: x in holiday_list
    )
    test_data[list_name] = test_data.loc[test_data['country'] == country, 'date'].apply(
        lambda x: x in holiday_list
    )

for i in countries:
    get_holiday_list(i, years)

#%% Create a boolean column indicating whether the date for a country was a holiday

def if_holiday(df):
    if df['Argentina_holidays'] is True or df['Canada_holidays'] is True or \
        df['Estonia_holidays'] is True or df['Japan_holidays'] is True or \
            df['Spain_holidays'] is True:
        return True
    else:
        return False

train_data['is_holiday'] = train_data.iloc[:, -5:].apply(if_holiday, axis=1)
test_data['is_holiday'] = test_data.iloc[:, -5:].apply(if_holiday, axis=1)

train_data = train_data.drop(['Argentina_holidays', 'Canada_holidays', 'Estonia_holidays',
                             'Japan_holidays', 'Spain_holidays'], axis=1)
test_data = test_data.drop(['Argentina_holidays', 'Canada_holidays', 'Estonia_holidays',
                             'Japan_holidays', 'Spain_holidays'], axis=1)

