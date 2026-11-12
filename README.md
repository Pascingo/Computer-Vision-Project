# Solarpotenzial-Karte Schweiz

Erkennung von Photovoltaik-Anlagen auf Schweizer Dächern mit Computer Vision, kombiniert mit dem Solarkataster des Bundes. Das Ergebnis ist eine Karte, die pro Dach zeigt, ob es für Solarenergie gut geeignet ist, ob bereits eine Anlage montiert ist und wie viel Potenzial noch frei ist.

Modularbeit Computer Vision

## Idee

Der Bund weiss dank [Sonnendach.ch](https://www.sonnendach.ch), welche Dächer sich für Solarenergie eignen. Welche davon bereits belegt sind, ist aber nicht flächendeckend bekannt: Das BFE-Anlagenregister führt Anlagen nur als Punkt mit Leistung, ohne die belegte Dachfläche.

Wir schliessen diese Lücke mit einem selbst trainierten Segmentierungsmodell, das Solaranlagen auf Luftbildern erkennt. Die erkannten Flächen verschneiden wir mit den Dachflächen aus Sonnendach.ch und berechnen so pro Dach den Belegungsgrad und das freie Potenzial.

## Forschungsfrage

Wie zuverlässig erkennt ein von Grund auf trainiertes Modell, das nur auf französischen Luftbildern gelernt hat, Solaranlagen auf Schweizer Dächern, und wie unterscheidet sich die Erkennung zwischen einer Gemeinde in der Westschweiz und einer in der Ostschweiz?

## Vorgehen

1. **Training:** Ein U-Net wird in PyTorch selbst implementiert und ohne Vortraining auf dem französischen Datensatz BDAPPV trainiert.
2. **Testgemeinden:** Das Modell wird auf zwei Schweizer Gemeinden angewendet:
   - eine kleine Gemeinde in der Waadt Richtung Genf (baulich näher an den französischen Trainingsdaten)
   - eine ländliche Gemeinde in der Ostschweiz (Thurgau)
3. **Evaluation:** Für jede Gemeinde wird ein eigenes Testset von Hand gelabelt. Zusätzlich gleichen wir die Erkennungen mit dem BFE-Anlagenregister ab.
4. **Verschneidung:** Erkannte Anlagen werden mit den Sonnendach-Dachflächen kombiniert. Belegungsgrad = PV-Fläche / Dachfläche.
5. **Karte:** Darstellung des freien Potenzials pro gut oder sehr gut geeignetem Dach.

### Vergleichsexperiment

| Variante | Training | Frage |
| --- | --- | --- |
| U-Net von Grund auf (Kernmodell) | BDAPPV, zufällige Initialisierung | Wie gut überträgt sich ein selbst gelerntes Modell auf die Schweiz? |
| U-Net mit ImageNet-Encoder | BDAPPV, vortrainierter Encoder | Wie viel bringt Vortraining? |
| Kernmodell + Nachtraining | zusätzlich wenige Schweizer Kacheln | Wie stark ist der Domain Shift und wie billig lässt er sich beheben? |

### Metriken

- **Pixel-Ebene:** IoU und F1, getrennt für BDAPPV und beide Testgemeinden
- **Gebäude-Ebene:** Precision und Recall gegenüber dem BFE-Anlagenregister

## Daten

| Datensatz | Quelle | Verwendung |
| --- | --- | --- |
| [BDAPPV](https://zenodo.org/records/7358126) | Mines Paris, RTE | Training und Validierung (Luftbilder + Segmentierungsmasken von PV-Anlagen in Frankreich) |
| SWISSIMAGE 10 cm | swisstopo | Luftbilder der Testgemeinden |
| [Solarenergie: Eignung Dächer](https://opendata.swiss/de/dataset/eignung-von-hausdachern-fur-die-nutzung-von-sonnenenergie) | BFE (Sonnendach.ch) | Dachflächen mit Einstrahlung und Eignungsklasse |
| [Elektrizitätsproduktionsanlagen](https://opendata.swiss/de/dataset/elektrizitatsproduktionsanlagen) | BFE | Validierung auf Gebäudeebene, Auswahl von Kacheln zum Labeln |
| Eigene Labels | Projektteam | Testsets für die beiden Gemeinden |

Rohdaten werden nicht ins Repository eingecheckt (siehe `data/README.md` für Download-Anleitungen).

## Geplante Struktur

```
├── data/
│   ├── raw/            # heruntergeladene Rohdaten (nicht versioniert)
│   ├── processed/      # Kacheln, Masken, Splits
│   └── labels/         # eigene Annotationen der Testgemeinden
├── notebooks/          # Exploration und Auswertung
├── src/
│   ├── data/           # Download, Kachelung, Auflösungsangleichung
│   ├── models/         # U-Net-Implementierung
│   ├── training/       # Trainingsschleife, Verlustfunktionen, Augmentation
│   ├── evaluation/     # Metriken, Abgleich mit Anlagenregister
│   └── mapping/        # Verschneidung mit Sonnendach, Kartenerstellung
├── results/            # Modelle, Metriken, Karten
└── README.md
```

## Tech-Stack

Python, PyTorch, GeoPandas, Rasterio, Label Studio oder CVAT für die Annotation.

## Bekannte Einschränkungen

- Luftbilder sind je nach Region einige Jahre alt; neuere Anlagen fehlen darauf.
- Photovoltaik und Solarthermie sehen sich auf Luftbildern ähnlich und werden vorerst als eine Klasse behandelt.
- Das Projekt beschränkt sich auf zwei Testgemeinden, nicht auf die ganze Schweiz.

## Verwandte Arbeiten

- [SolAI, FHNW](https://www.fhnw.ch/de/architektur-bau-geomatik/forschung-dienstleistungen/forschung/projekte/solai): Erkennung von Solaranlagen auf SWISSIMAGE
- [Castello et al. 2019, EPFL](https://infoscience.epfl.ch/record/272682): U-Net für Dach-PV auf Schweizer Luftbildern
- [Baenziger 2022, ZHAW](https://digitalcollection.zhaw.ch/bitstream/11475/27103/1/2022_Tim_Baenziger_MSc_WI.pdf): Mask R-CNN mit SWISSIMAGE und Sonnendach
- [Kasmi et al. 2023](https://www.nature.com/articles/s41597-023-01951-4): BDAPPV-Datensatz

## Team

- Timon Reichmuth 
- Pascal Ingold
