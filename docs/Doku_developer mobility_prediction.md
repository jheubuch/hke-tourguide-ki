# Developer-Doku mobility_prediction

Im Folgenden wird die Verwendung der von `SinglePredict()` und `MultiPredict()` verwendeten Methoden aus `mobility_prediction.py` zur Vorhersage von Besucherzahlen für Landkreise und Zeiträume beschrieben.
<br>
<br>

# Methoden

## getPredictionDataStructure(numberOfRows)

### Funktion

### Parameter

| Parameter | Typ | Beschreibung |
| --------- | --- | ------------ |
| ``        | ``  | Bla          |
| ``        | ``  | Bla          |

### Rückgabe

## getReturnDataStructure(numberOfRows)

### Funktion

### Parameter

| Parameter | Typ | Beschreibung |
| --------- | --- | ------------ |
| ``        | ``  | Bla          |
| ``        | ``  | Bla          |

### Rückgabe

## getSavedDistricts()

### Funktion

### Parameter

### Rückgabe

## addPredictionDataset(dtf, predictions, districts, districtId, date, datasetIndex)

### Funktion

Befüllt die DataFrames, für die eine Prognose erstellt werden soll, mit den zur
Prognose benötigten Daten.

### Parameter

| Parameter | Typ | Beschreibung |
| --------- | --- | ------------ |
| `dtf`        | `DataFrame`  | DataFrame, der alle Daten enthalten soll, die für die Prognose relevant sind. |
| `predictions`        | `DataFrame`  | DataFrame, der die Prognoseergebnisse enthalten soll |
| `disctricts`        | `DataFrame`  | DataFrame mit allen Landkreisen |
| `districtId`        | `string`  | Landkreis-ID für den die Prognose erstellt werden soll |
| `date`        | `string`  | Datum, für das die Prognose erstellt werden soll |
| `datasetIndex`        | `int`  | Offset, Nummer der Prognose (Standard: `0`) |

### Rückgabe

Tupel mit gefüllten DataFrames `dtf, predictions`

## predict(dtf)

### Funktion

Die Funktion `predict` erstellt die tatsächliche Prognose aufgrund der gegebenen
Daten. Hierfür wird das trainierte Model geladen und eine Vorhersage mittels
gegebener Daten getroffen.

### Parameter

| Parameter | Typ | Beschreibung |
| --------- | --- | ------------ |
| `dtf`        | `DataFrame`  | Dataframe mit allen, zur Vorhersage benötigten Daten |

### Rückgabe

Es wird ein `int` zurückgegeben. Dieser Wert entspricht der prognostizierten
Besucherzahl.