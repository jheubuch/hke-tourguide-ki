import pandas as pd
import numpy as np
import os.path
import datetime
import pickle
import meteostat as met
import requests


def isTravellingDate(date):
    travellingDates = getTravellingDates(datetime.datetime.strptime(date, "%Y-%m-%d").year)
    return date in travellingDates

def isWorkDay(datestring):
    freeDays =  ["Saturday", "Sunday"]
    return not (datetime.datetime.strptime(datestring, "%Y-%m-%d").strftime("%A") in freeDays)

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

def getWeatherForDateAndPoint(date, point, desired_data = ['tmax', 'prcp']):
    weather_data = met.Daily(point, date, date).fetch()
    weather_dict = {}
    if(weather_data.empty): return None
    for data_col in desired_data:
        weather_dict[data_col] = weather_data.iloc[0][data_col]

    return weather_dict

def getPredictionDataStructure(numberOfRows):
    # Load csv with data structure
    dataFileName ="../data/mobilityData_complete.csv"
    if os.path.isfile(dataFileName):
        dtf = pd.read_csv(dataFileName)
    else:
        print("Data file could not be loaded")
        return

    # Extract column names from data structure
    doWColumns = [col for col in dtf if col.startswith('DoW_')]
    dayColumns = [col for col in dtf if col.startswith('Day_')]
    weatherColumns = ["MaxTemp", "Precip"]
    endIdColumns = [col for col in dtf if col.startswith('EndId_')]
    columns = doWColumns + dayColumns + endIdColumns + weatherColumns

    # Create dtf with n rows
    data = np.array([np.zeros(numberOfRows, dtype=np.int32)]*columns.__len__()).T

    dtf = pd.DataFrame(data, columns=columns)
    dtf.fillna(0)

    return dtf

def getReturnDataStructure(numberOfRows):
    cols = ["Date", "DistrictId", "DistrictName", "MaxTemp", "Precip", "Visitors"]
    # Create dtf with single row
    data = np.array([np.zeros(numberOfRows, dtype=np.int32)]*cols.__len__()).T
    return pd.DataFrame(data, columns=cols)

def getDatesToPredict(startDate, endDate):
    now = datetime.datetime.today()
    startDate = datetime.datetime.strptime(startDate, "%Y-%m-%d")
    endDate = datetime.datetime.strptime(endDate, "%Y-%m-%d")

    intoFuture = endDate - now
    if(intoFuture.days > 10):
        print("Preditions more than 10 days into the future are too unreliable")
        return

    delta = endDate - startDate

    datestrings = []

    for i in range(delta.days + 1):
        day = startDate + datetime.timedelta(days=i)
        datestrings.append(day.strftime("%Y-%m-%d"))

    return datestrings

def getSavedDistricts():
     # Load districts
    districtsFileName = "../data/districts.csv"
    if os.path.isfile(districtsFileName):
        # load saved model
        return pd.read_csv("../data/districts.csv")
    else:
        print("No saved districts could be loaded")
        return None

def addPredictionDataset(dtf, predictions, districts, districtId, date, datasetIndex=0):
    predictions.loc[datasetIndex, 'Date'] = date
    predictions.loc[datasetIndex, 'DistrictId'] = districtId
    predictions.loc[datasetIndex, 'DistrictName'] = districts.loc[(districts.EndId == districtId), 'EndName'].values[0]

    # Fill in EndId and date values
    endIdColumn = 'EndId_' + str(districtId)
    dtf.loc[datasetIndex, endIdColumn] = 1
    
    # Generate datetime object from passed date
    dateObject = datetime.datetime.strptime(date, "%Y-%m-%d")

    # Fill in DoW type  
    dtf.loc[datasetIndex, "DoW_" + dateObject.strftime("%A")] = 1

    # Fill in Day_class_work
    dtf.loc[datasetIndex, "Day_class_work"] = int(isWorkDay(date))

    # Fill in Day_class_travel type
    dtf.loc[datasetIndex, "Day_class_travel"] = int(isTravellingDate(date))

    # print(dtf[["MaxTemp", "Precip"]].head())

    # Get and add weather data
    weatherData = getWeatherForDateAndPoint(dateObject, met.Point(districts.loc[(districts.EndId == districtId), 'Lat'].values[0], districts.loc[(districts.EndId == districtId), 'Lon'].values[0]))
    if(weatherData == None): return dtf, predictions
    dtf.loc[datasetIndex, "MaxTemp"] = weatherData["tmax"]
    dtf.loc[datasetIndex, "Precip"] = weatherData["prcp"]
    predictions.loc[datasetIndex, 'MaxTemp'] = weatherData["tmax"]
    predictions.loc[datasetIndex, "Precip"] = weatherData["prcp"]

    return dtf, predictions

def predict(dtf):
    modelFileName = "mobilityPredictionModel.sav"
    if os.path.isfile(modelFileName):
        # load saved model
        model = pickle.load(open(modelFileName, 'rb'))
    else:
        print("No saved model could be loaded")
        return

    return np.int32(np.round(model.predict(dtf.values), decimals=0))
    
def singlePredict(districtId, date):
    dtf = getPredictionDataStructure(1)
    predictions = getReturnDataStructure(1)
    districts = getSavedDistricts()
    if(districts.empty): return
    dtf, predictions = addPredictionDataset(dtf, predictions, districts, districtId, date)
    predictions['Visitors'] = predict(dtf)
    return predictions

def multiPredict(districtIds, startdate, enddate):
    districts = getSavedDistricts()
    if(districts.empty): return
    dates = getDatesToPredict(startdate, enddate)
    predictions = getReturnDataStructure(len(districtIds)*len(dates))
    dtf = getPredictionDataStructure(len(districtIds)*len(dates))
    for i, districtId in enumerate(districtIds):
        for j, date in enumerate(dates):
            dtf, predictions = addPredictionDataset(dtf, predictions, districts, districtId, date, i*len(dates)+j)
    predictions['Visitors'] = predict(dtf)
    return predictions