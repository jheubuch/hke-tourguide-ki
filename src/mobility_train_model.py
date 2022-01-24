# for data
from statsmodels.graphics.api import abline_plot
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# for statistical tests
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from sklearn.ensemble import GradientBoostingClassifier

# for model persistence
import pickle

# for files
import os.path

# for date
import datetime

# for HTTP
import requests

# for weather
import meteostat as met
from geopy.geocoders import Nominatim

# for dataframe nesting
from weatherData import WeatherData 


geolocator = Nominatim(user_agent="hke-tgki/0.1")

def utils_recognize_type(dtf, col, max_cat=20):
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"

def prepareDtf(dtf):
    dtf.columns = dtf.columns.str.strip()
    cols = ["Bucket", "StartId", "StartName", "EndId", "EndName", "Count"]

    dtf = dtf[cols]
    dtf = dtf[dtf['Bucket'].str.len() == 10]
    dtf = dtf.groupby(["Bucket", "EndId", "EndName"])[
        "Count"].sum().reset_index()

    dtf["DoW"] = dtf["Bucket"].apply(
        lambda bucket: pd.to_datetime(bucket).day_name())

    dtf = dtf[dtf['Bucket'] >= '2020-04-01']

    dtf = dtf.rename(columns={"Count": "Y"})
  
    return dtf

def prepareDistricts(dtf):
    districts = dtf[['EndId', 'EndName']].copy()
    districts.drop_duplicates(subset = ["EndId"], inplace=True)
    # districts["Point"] = ""
    districts["Lat"] = ""
    districts["Lon"] = ""
    districts["Weather"] = ""
    for i in districts.index:
        lat, lon = getCoordsForDistrict(districts['EndName'][i])
        if (lat == None or lon == None): continue
        # districts.loc[i,'Point'] = point
        districts.loc[i,'Lat'] = lat
        districts.loc[i,'Lon'] = lon
    
    return districts

def removeOutliers(dtf, factor): 
    maxVisitors = dtf.sort_values(by=['Y'])
    q1 = np.quantile(maxVisitors['Y'], 0.50)
    q3 = np.quantile(maxVisitors['Y'], 0.90)
    iqr = q3 - q1

    minValue = q1 - factor*iqr
    maxValue = q3 + factor*iqr

    print("minValue: ", minValue, "\nmaxValue: ", maxValue)

    dtf = dtf[dtf['Y'] >= minValue]
    dtf = dtf[dtf['Y'] <= maxValue]
    return dtf
    
def printHeatmap(dtf):
    dic_cols = {col: utils_recognize_type(
        dtf, col, max_cat=20) for col in dtf.columns}
    heatmap = dtf.isnull()
    for k, v in dic_cols.items():
        if v == "num":
            heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
        else:
            heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
    sns.heatmap(heatmap, cbar=False).set_title('Dataset Overview')
    plt.show()
    print("\033[1;37;40m Categerocial ",
          "\033[1;30;41m Numeric ", "\033[1;30;47m NaN ")

def showKfoldValidation(model, x_train, y_train):
    scores = []
    cv = model_selection.KFold(n_splits=5, shuffle=True)
    fig = plt.figure()
    i = 1
    for train, test in cv.split(x_train, y_train):
        prediction = model.fit(x_train[train],
                               y_train[train]).predict(x_train[test])
        true = y_train[test]
        score = metrics.r2_score(true, prediction)
        scores.append(score)
        plt.scatter(prediction, true, lw=2, alpha=0.3,
                    label='Fold %d (R2 = %0.2f)' % (i, score))
        i = i+1
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)],
             linestyle='--', lw=2, color='black')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('K-Fold Validation')
    plt.legend()
    plt.show()

def scaleData(dtf, y):
    # scale X
    scalerX = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
    X = scalerX.fit_transform(dtf.drop("Y", axis=1))
    dtf_scaled = pd.DataFrame(X, columns=dtf.drop(
        y, axis=1).columns, index=dtf.index)

    # scale Y
    scalerY = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
    dtf_scaled[y] = scalerY.fit_transform(
        dtf[y].values.reshape(-1, 1))

    return dtf_scaled, scalerX, scalerY

def unscaleData(scalerY, predicted, y_test):
    predicted = scalerY.inverse_transform(
        predicted.reshape(-1, 1)).reshape(-1)
    y_test = scalerY.inverse_transform(
        y_test.reshape(-1, 1)).reshape(-1)
    return predicted, y_test

def printKPI(predicted, y_test):
    # print("R2 (explained variance):", round(
    #     metrics.r2_score(y_test, predicted), 3) * 100, "%")
    # print("Mean Absolute Perc Error (Σ(|y-pred|/y)/n):",
    #       round(np.mean(np.abs((y_test-predicted)/predicted)), 3) * 100, "%")
    print("Mean Absolute Error (Σ|y-pred|/n):",
          "{:,.0f}".format(metrics.mean_absolute_error(y_test, predicted)))
    print("Root Mean Squared Error (sqrt(Σ(y-pred)^2/n)):",
          "{:,.0f}".format(np.sqrt(metrics.mean_squared_error(Y_test, predicted))))
    print("95-percentile Mean Absolute Error:", 
          "{:,.0f}".format(round(np.mean(stats.trim1(np.abs((y_test-predicted)),0.05, "right")), 0)))

def computeResiduals(predicted, y_test):
    residuals = y_test - predicted
    max_error = max(residuals) if abs(max(residuals)) > abs(
        min(residuals)) else min(residuals)
    max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(
        min(residuals)) else list(residuals).index(min(residuals))
    max_true, max_pred = y_test[max_idx], predicted[max_idx]
    print("Max Error:", "{:,.0f}".format(max_error))
    return residuals, max_error, max_idx, max_true, max_pred

def printMaxErrorY(predicted, y_test):
    max_difference_y = 0
    max_dif = 0
    for idx, val in enumerate(y_test):
        dif = abs(val - predicted[idx])
        if(dif > max_dif):
            max_dif = dif
            max_difference_y = val
    print("Max Error Dataset:")
    print(dtf.loc[dtf['Y'] == max_difference_y, ["Bucket", "EndId", "Y"]])

def plotPredictionKPI(predicted, y_test, max_error, max_true, max_pred):
    # Plot predicted vs true
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].scatter(predicted, Y_test, color="black")
    abline_plot(intercept=0, slope=1, color="red", ax=ax[0])
    ax[0].vlines(x=max_pred, ymin=max_true, ymax=max_true-max_error,
                 color='red', linestyle='--', alpha=0.7, label="max error")
    ax[0].grid(True)
    ax[0].set(xlabel="Predicted", ylabel="True", title="Predicted vs True")
    ax[0].legend()

    # Plot predicted vs residuals
    ax[1].scatter(predicted, residuals, color="red")
    ax[1].vlines(x=max_pred, ymin=0, ymax=max_error, color='black',
                 linestyle='--', alpha=0.7, label="max error")
    ax[1].grid(True)
    ax[1].set(xlabel="Predicted", ylabel="Residuals",
              title="Predicted vs Residuals")
    ax[1].hlines(y=0, xmin=np.min(predicted), xmax=np.max(predicted))
    ax[1].legend()
    plt.show()

def plotResidualDistribution(residuals):
    fig, ax = plt.subplots()
    sns.distplot(residuals, color="red", hist=True,
                 kde=True, kde_kws={"shade": True}, ax=ax)
    ax.grid(True)
    ax.set(yticks=[], yticklabels=[], title="Residuals distribution")
    plt.show()

def getFreeDays(year):
    # Request free days from Feiertage API for year 2021 in bavaria
    free_days = requests.get(
        f'https://feiertage-api.de/api/?jahr={year}&nur_land=BY')

    # Parse result and store as [key, val] dict
    free_days = free_days.json()

    free_day_dates = []

    # Iterate over
    for free_day_name in free_days:
        free_day_dates.append(free_days[free_day_name]['datum'])

    return free_day_dates

def getHolidays(year):
    # Request free days from Feiertage API for year 2021 in bavaria
    holidays = requests.get(
    f'https://ferien-api.de/api/v1/holidays/BY/{year}')

    # Parse result and store as [key, val] dict
    holidays = holidays.json()

    holiday_dates = []

    # Iterate over
    for holiday in holidays:
        holiday_dates.append((holiday['start'], holiday['end']))

    return holiday_dates

def getCoordsForDistrict(districtName):
    location = geolocator.geocode(districtName)
    if (location == None):
        print(districtName)
        return None, None
    return location.latitude, location.longitude

def getTravellingDates(year, timespan=4): 
    holiday_dates = getHolidays(year)

    travelling_datestrings = []

    for date in holiday_dates:
        start = datetime.datetime.strptime(date[0], "%Y-%m-%dT%H:%MZ")
        end = datetime.datetime.strptime(date[1], "%Y-%m-%dT%H:%MZ")
        for i in range(0, timespan-1, 1):
            travelling_datestrings.append(datetime.datetime.strftime(start + datetime.timedelta(days=i), "%Y-%m-%d"))
            travelling_datestrings.append(datetime.datetime.strftime(end - datetime.timedelta(days=i),"%Y-%m-%d" ))

    return travelling_datestrings



def getWeatherData(dtf, districts):
    # Set time period
    start = datetime.datetime.strptime(dtf.iloc[1]['Bucket'], "%Y-%m-%d")
    end = datetime.datetime.strptime(dtf.iloc[-1]['Bucket'], "%Y-%m-%d")

    for i in districts.index:
        districts.loc[i,'Weather'] = WeatherData(met.Daily(met.Point(districts['Lat'][i], districts['Lon'][i]), start, end).fetch())


def addWeatherData(dtf, districts): 
    dtf["MaxTemp"] = ""
    dtf["Precip"] = ""

    for i in dtf.index:
        id = dtf.loc[i, 'EndId']
        time = dtf.loc[i, 'Bucket']
        weatherData = districts.loc[(districts.EndId == id),'Weather'].values[0]

        if(weatherData.data.loc[(weatherData.data.index == pd.Timestamp(time))].size < 1):
            continue

        dtf.loc[i, 'MaxTemp'] = weatherData.data.loc[(weatherData.data.index == pd.Timestamp(time)), 'tmax'].values[0]
        dtf.loc[i, 'Precip'] = weatherData.data.loc[(weatherData.data.index == pd.Timestamp(time)), 'prcp'].values[0]


print("Starting")
dtf = pd.read_csv("../data/mobilityData.csv")
dtf = prepareDtf(dtf)
print("Completed mobility data loading")


districts = prepareDistricts(dtf)
districts.to_csv("../data/districts.csv")
print("Completed districts preparation")



# Load saved districts
districts = pd.read_csv("../data/districts.csv")
districts.head()


getWeatherData(dtf, districts)
addWeatherData(dtf, districts)
dtf.to_csv("../data/mobilityData_extended_weather.csv")
print("Completed weather data preparation")



# Remove invalid values
print("Rows before NaN removal: ", dtf.shape[0])
dtf.replace([np.inf, -np.inf], np.NaN, inplace=True)
dtf.replace("", np.NaN, inplace=True)
dtf.dropna(how='any', axis=0, inplace=True)
print("Rows after NaN removal: ",dtf.shape[0])


# define clusters (workingday/weekend)
x = "Day"
Day_clusters = {"free": ["Saturday", "Sunday"], "work": ["Monday", "Tuesday", "Wednesday",
                                                        "Thursday", "Friday"]}
# create DoW_class columns
dic_flat = {v: k for k, lst in Day_clusters.items() for v in lst}
for k, v in Day_clusters.items():
    if len(v) == 0:
        residual_class = k
dtf[x+"_class"] = dtf['DoW'].apply(lambda x: dic_flat[x] if x in
                                dic_flat.keys() else residual_class)

# create dummies Day_class
dummy = pd.get_dummies(dtf["Day_class"],
                    prefix="Day_class", drop_first=True)
dtf = pd.concat([dtf, dummy], axis=1)
freeDays = getFreeDays('2020')
for day in freeDays:
    dtf.loc[(dtf.Bucket == day), 'Day_class_work'] = 0

travellingDays = getTravellingDates('2020')
dtf['Day_class_travel'] = 0
for day in travellingDays:
    dtf.loc[(dtf.Bucket == day), 'Day_class_travel'] = 1
    
# drop the original column
dtf = dtf.drop("Day_class", axis=1)

dtf.head()


# create dummies DoW
dummy = pd.get_dummies(dtf["DoW"],
                    prefix="DoW", drop_first=False)
dtf = pd.concat([dtf, dummy], axis=1)
# drop the original column
dtf = dtf.drop("DoW", axis=1)


# create dummies EndId
dummy = pd.get_dummies(dtf["EndId"],
                    prefix="EndId", drop_first=False)
dtf = pd.concat([dtf, dummy], axis=1)

print("Completed dataset preparation")



# Save extended data
dtf.to_csv("../data/mobilityData_complete.csv")



# Load extended data
dtf = pd.read_csv("../data/data/mobilityData_complete.csv")
dtf.head()


# define training and test features
DoWColumns = [col for col in dtf if col.startswith('DoW_')]
DayColumns = [col for col in dtf if col.startswith('Day_')]
WeatherColumns = ["MaxTemp", "Precip"]
EndIdColumns = [col for col in dtf if col.startswith('EndId_')]
X_names = DoWColumns + DayColumns + EndIdColumns + WeatherColumns

relevantDtf = dtf[X_names + ["Y"]]

# split data
dtfTrain, dtfTest = model_selection.train_test_split(relevantDtf,
                                                    test_size=0.2)

X_train = dtfTrain[X_names].values
Y_train = dtfTrain["Y"].values
X_test = dtfTest[X_names].values
Y_test = dtfTest["Y"].values

print("Completed feature preparation")


# create model
model = ensemble.GradientBoostingRegressor(n_estimators=400, random_state=21)

# train
model.fit(X_train, Y_train)

# save model
pickle.dump(model, open("mobilityPredictionModel.sav", 'wb'))

print("Completed model training")


# test
predicted = model.predict(X_test)


printKPI(predicted, Y_test)

# residuals
residuals, max_error, max_idx, max_true, max_pred = computeResiduals(
    predicted, Y_test)

printMaxErrorY(predicted, Y_test)

plotPredictionKPI(predicted, Y_test, max_error, max_true, max_pred)
