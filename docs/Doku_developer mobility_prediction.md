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

Befüllt den DataFrame, für den eine Prognose erstellt werden soll, mit den zur
Prognose benötigten Daten. Außerdem werden für die Prognose relevante Ausgabewerte direkt in den Prognosedataframe übernommen.

### Parameter

| Parameter      | Typ         | Beschreibung                                                                                                        |
| -------------- | ----------- | ------------------------------------------------------------------------------------------------------------------- |
| `dtf`          | `DataFrame` | DataFrame, der alle Daten enthalten soll, die für die Prognose relevant sind.                                       |
| `predictions`  | `DataFrame` | DataFrame, der die Prognoseergebnisse enthalten soll                                                                |
| `disctricts`   | `DataFrame` | DataFrame mit allen Landkreisen                                                                                     |
| `districtId`   | `string`    | Landkreis-ID für den die Prognose erstellt werden soll                                                              |
| `date`         | `string`    | Datum, für das die Prognose erstellt werden soll                                                                    |
| `datasetIndex` | `int`       | Zeile des Dataframes, in welche die Daten eingetragen werden sollen, entspricht Nummer der Prognose (Standard: `0`) |

### Rückgabe

Tupel mit gefüllten DataFrames `dtf, predictions`

## predict(dtf)

### Funktion

Erstellt die tatsächliche Prognose aufgrund der gegebenen
Daten. Hierfür wird das trainierte Model geladen und eine Vorhersage mittels
gegebener Daten getroffen.

### Parameter

| Parameter | Typ         | Beschreibung                                         |
| --------- | ----------- | ---------------------------------------------------- |
| `dtf`     | `DataFrame` | Dataframe mit allen, zur Vorhersage benötigten Daten |

### Rückgabe

Es wird ein `int` zurückgegeben. Dieser Wert entspricht der prognostizierten
Besucherzahl.
