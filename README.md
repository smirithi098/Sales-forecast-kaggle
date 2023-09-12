# Sales Forecast - Kaggle 

Link to project: [Project on kaggle](https://www.kaggle.com/code/smirithi/sales-forecast)

The objective of the project is to predict the sales of various fictious learning modules curated by kaggle in different countries sold by various stores in the year 2022. The training data ranges from January, 2017 to December, 2021.

### Data Preparation
As we have the *Date* columns, we can extract other useful information from this feature namely:
- Day of the week (1-7)
- Month (1-12)
- is the day a *weekend*
- is it a *holiday* as per the country's holiday list

Also, the text columns have been converted to numeric ones using a **Label encoder** to facilitate smooth model training.

### EDA
From the exploratory data analysis, we have found following insights:
1. The sales is at peek during the winters i.e. in the months of *December & January*.
2. Weekday seems to have a higher number of modules sold in comparison to the weekends.
3. In almost all countries, the sales saw a slight rise during the holiday season.

### Models Applied
**RandomForestRegressor** was used to predict the sale numbers for the next year. Firstly, the train data was split into redictors and response variable. Furthermore, the data was split into train and test data to prevent over-fitting or under-fitting. 
The *Mean absolute error* on the test data was at 115 and the model managed to predict most of the numbers quiet accurately. The performance can be improved by implementing time series analysis/neural networks.

### Packages Used
- Pandas
- Numpy
- Datetime
- Sklearn
- Seaborn
