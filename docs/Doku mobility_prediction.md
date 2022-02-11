# Doku mobility_prediction

Im Folgenden wird die Verwendung der relevanten Methoden aus `mobility_prediction.py` zur Vorhersage von Besucherzahlen für Landkreise und Zeiträume beschrieben.
<br>
<br>

# Voraussetzungen

- `mobility_train_model.py` muss mindestens ein Mal erfolgreich ausgeführt worden sein. Hierbei werden für die Vorhersage wichtige Dateien generiert, welche im folgenden aufgeführt sind.
- `mobilityPredictionModel.sav` muss im aktuellen Arbeitsverzeichnis vorhanden sein
- `districts.csv` muss im Verzeichnis `../data` relativ zum Arbeitsverzeichnis vorhanden sein
- `mobilityData_complete.csv` muss im Verzeichnis `../data` relativ zum Arbeitsverzeichnis vorhanden sein
- Es muss eine Internetverbindung zur Abfrage von aktuellen Wetterdaten vorhanden sein

<br>

# Methoden

## singlePredict(districtId, date)

Liefert für eine übergebene `districtId` und ein `date` eine Besucherzahlenprognose für den jeweiligen Landkreis am entsprechenden Tag.
Der abgefragte Tag darf nicht mehr als 10 Tage in der Zukunft liegen, da sonst die entsprechenden Wetterdaten zu unzuverlässig wären.
<br>
<br>

| Parameter    | Typ      | Beschreibung                                                                                                                           |
| ------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `districtId` | `int`    | Landkreis-ID des Landkreises für den eine Prognose erstellt werden soll. (Entspricht `EndId` bzw. `StartId` aus den Terralytics-Daten) |
| `date`       | `string` | Datum (`YYYY-MM-DD`) des Tages für den eine Pronose erstellt werden soll                                                               |

<br>

### Rückgabe

Pandas Dataframe mit einer einzelnen Zeile und folgenden Spalten:

| Date  | DistrictId   | DistrictName  | MaxTemp          | Precip             | Visitors       |
| ----- | ------------ | ------------- | ---------------- | ------------------ | -------------- |
| Datum | Landkreis-ID | Landkreisname | Höchsttemperatur | Niederschlagsmenge | Besucheranzahl |

<br>
<br>
<br>

## multiPredict(districtIds, startdate, enddate)

Liefert für eine oder mehrere übergebene `districtIds` jeweils eine Vorhersage für jeden Landkreis und Tag im Zeitraum von `startdate` (einschließlich) und `enddate` (einschließlich). Der abgefragte Zeitraum darf nicht mehr als 10 Tage in der Zukunft liegen, da sonst die entsprechenden Wetterdaten zu unzuverlässig wären.
<br>
<br>

| Parameter     | Typ      | Beschreibung                                                                                                                                    |
| ------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `districtIds` | `int[]`  | Liste von Landkreis-IDs der Landkreise für die Prognosen erstellt werden sollen. (Entsprechen `EndId` bzw. `StartId` aus den Terralytics-Daten) |
| `startdate`   | `string` | Datum (`YYYY-MM-DD`) des ersten Tages für den eine Pronose erstellt werden soll                                                                 |
| `enddate`     | `string` | Datum (`YYYY-MM-DD`) des letzten Tages für den eine Pronose erstellt werden soll                                                                |

<br>

### Rückgabe

Pandas Dataframe mit je einer Zeile pro Landkreis und Tag im Zeitraum und folgenden Spalten:

| Date  | DistrictId   | DistrictName  | MaxTemp          | Precip             | Visitors       |
| ----- | ------------ | ------------- | ---------------- | ------------------ | -------------- |
| Datum | Landkreis-ID | Landkreisname | Höchsttemperatur | Niederschlagsmenge | Besucheranzahl |

<br>
<br>
<br>
