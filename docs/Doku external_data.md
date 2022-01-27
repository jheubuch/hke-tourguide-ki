# Verwendung externer Daten

Das Mobilitätsverhalten von Personen ist von gewissen Einflussfaktoren
abhängig. Um die bestehenden Mobilitätsdaten mit diesen Faktoren zu
matchen, sind externe Datenquellen nötig, von denen diese Daten bezogen
werden können.

## Feiertage

Da an Feiertagen wesentlich mehr Ausflugsverkehr herrscht, als an
anderen Tagen, ist es sinnvoll, diese Daten mit in den Lernprozess
aufzunehmen.

Für das Projekt wurde die REST-Schnittstelle von
[Feiertage-API](https://feiertage-api.de) verwendet. Diese liefert
Feiertagsdaten unter Angabe des gewünschten Jahres sowie optional eines
bestimmten Bundeslandes. In diesem Fall wurden für das Training des
Modells die Feiertagsdaten aus Bayern für das Jahr 2020 verwendet.
Abrufbar sind diese mittels eines HTTP-Requests mit einer Antwort im
JSON-Format.

## Ferienzeiten

Auch rund um Ferienzeiträume wächst das Mobilitätsverhalten von
Menschen stark an. So bietet es sich an, diese Zeiträume als
Reisezeiträume zu deklarieren und das Modell darauf basierend zu
trainieren.

Für das Projekt wurde die REST-Schnittstelle von
[Ferien-API](https://ferien-api.de) verwendet. Auch mit dieser
Schnittstelle können Feriendaten für ein bestimmtes Jahr und ein
bestimmtes Bundesland abgerufen werden. Durch diese Schnittstelle
werden sowohl Start- als auch Enddaten bekannt und somit können
Reisezeiträume von anderen abgegrenzt werden. Auch dieses API
liefert die Daten per HTTP und JSON-Antwort.

## Wetter

Auch das zu erwartende Wetter ist ein wesentlicher Einflussfaktor,
da viele Personen spontane Reiseziele auch aufgrund des Wetters
aussuchen. Zur Ermittlung von Wetterdaten wurde die Python-Bibliothek
[Meteostat](https://dev.meteostat.net/python) verwendet.
Hier können Wetterdaten für die Vergangenheit und für die Zukunft
durch Angabe der entsprechenden Koordinaten angefragt werden.
Meteostat verwendet dann das "nearest neighbor"-Prinzip zur Ermittlung
der nächstgelegenen Wetterstation.

Da jedoch nur die Landkreisnamen vorliegen, muss vorher noch eine
Möglichkeit gefunden werden, anhand dieser Landkreisnamen automatisiert
Koordinaten für diese zu erhalten. Dies wurde mittels der Python-Bibliothek
[geopy](https://geopy.readthedocs.io/en/stable) umgesetzt. So kann
automatisiert für jeden Landkreis die entsprechenden Wetterdaten angefordert
werden.