# Developer-Doku mobility_prediction

Im Folgenden wird die Verwendung der von `SinglePredict()` und `MultiPredict()` verwendeten Methoden aus `mobility_prediction.py` zur Vorhersage von Besucherzahlen für Landkreise und Zeiträume beschrieben.
<br>
<br>

# Methoden

## getPredictionDataStructure(numberOfRows)

### Funktion

Generiert einen `Dataframe`, welcher die selben Spalten enthält wie der Dataframe welcher zum Training des Models verwendet wurde (was einem `Dataframe` entspricht, für den das Trainierte Model eine Prognose erstellen kann). Hierfür muss die Datei `../data/mobilityData_complete.csv` vorhanden sein, um die entsprechenden Spalten extrahieren zu können.

### Parameter

| Parameter      | Typ   | Beschreibung                                                                                                            |
| -------------- | ----- | ----------------------------------------------------------------------------------------------------------------------- |
| `numberOfRows` | `int` | Gibt an, wie viele Zeilen der zurückzugebende Dataframe erhalten soll (entspricht Anzahl der zu erstellenden Prognosen) |

### Rückgabe

`Dataframe` mit entsprechenden Spalten um Prognose durch Model erstellen zu lassen und `numberOfRows` Zeilen (alle Werte `= 0`)

## getReturnDataStructure(numberOfRows)

### Funktion

Generiert einen `Dataframe`, welcher die Spalten enthält die als Ausgabewerte einer Prognose relevant sind. Denkbar sind hierfür Werte wie Datum, Besucheranzahl, Landkreisname, Höchsttemperatur, Niederschlag etc.

### Parameter

| Parameter      | Typ   | Beschreibung                                                                                                            |
| -------------- | ----- | ----------------------------------------------------------------------------------------------------------------------- |
| `numberOfRows` | `int` | Gibt an, wie viele Zeilen der zurückzugebende Dataframe erhalten soll (entspricht Anzahl der zu erstellenden Prognosen) |

### Rückgabe

`Dataframe` mit entsprechenden Spalten für Informationen rund um die Prognose und `numberOfRows` Zeilen (alle Werte `= 0`)

## getSavedDistricts()

### Funktion

Lädt die gespeicherten Landkreisdaten aus der Datei `../data/districts.csv` in einen `Dataframe`.

### Rückgabe

`Dataframe` mit gespeicherten Landkreisdaten.

## addPredictionDataset(dtf, predictions, districts, districtId, date, datasetIndex)

### Funktion

### Parameter

| Parameter | Typ | Beschreibung |
| --------- | --- | ------------ |
| ``        | ``  | Bla          |
| ``        | ``  | Bla          |

### Rückgabe

## predict(dtf)

### Funktion

### Parameter

| Parameter | Typ | Beschreibung |
| --------- | --- | ------------ |
| ``        | ``  | Bla          |
| ``        | ``  | Bla          |

### Rückgabe
