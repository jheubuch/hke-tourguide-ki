# Doku mobility_train_model

Im Folgenden wird die Verwendung von `mobility_train_model.py` zum Training eines Vorhersagemodells von Besucherzahlen für Landkreise und Zeiträume beschrieben.
<br>
<br>

# Voraussetzungen

- Es muss eine Internetverbindung zur Abfrage von zusätzlichen Daten vorhanden sein
- Die verwendeten Mobilitätsdaten sollten nicht aus mehreren Kalenderjahren stammen
- `mobilityData.csv`, welche die zugrundeliegenden Mobilitätsdaten enthält, muss im Verzeichnis `../data` relativ zum Arbeitsverzeichnis vorhanden sein. Diese sollte (wie beim Terralytics Datensatz) mindestens folgende Spalten aufweisen:

| Bucket | StartId           | StartName          | EndId           | EndName          | Count          |
| ------ | ----------------- | ------------------ | --------------- | ---------------- | -------------- |
| Datum  | Startlandkreis-ID | Startlandkreisname | Endlandkreis-ID | Endlandkreisname | Besucheranzahl |

<br>

# Verwendung

Sind alle Voraussetzungen gegeben, so muss die Datei `mobility_train_model.py` lediglich ausgeführt werden. Nach der erfolgreichen Ausführung wird ein vollständig trainiertes machine learning model exportiert (`mobilityPredictionModel.sav`), welches anschließend für entsprechende Prognosen verwendet werden kann.
<br>
<br>

# Generierte Dateien

| Dateiname                     | Verzeichnis (rel. zu Arbeitsverzeichnis) | Funktion                                                                                                                                           |
| ----------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mobilityPredictionModel.sav` | `./`                                     | Export des trainierten machine learning models. Kann im Anschluss für Besucherzahlenprognose verwendet werden (bspw. von `mobility_prediction.py`) |
| `districts.csv`               | `../data`                                | Enthält gesammelte Daten zu den in den Mobilitätsdaten auftretenden Landkreisen                                                                    |
| `mobilityData_complete.csv`   | `../data`                                | Enthält aufbereitete, um externe Daten ergänzte Mobilitätsdaten in ihrer finalen Datenstruktur                                                     |
