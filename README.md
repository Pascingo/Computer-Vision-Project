# Detecting Existing Solar Installations on Rooftops Using Satellite Imagery

**Authors:** Pascal Ingold, Timon Reichmuth
Computer Vision module project, HSLU

---

## 1. Topic Description

### Project Idea
In this project we want to build a tool that analyses satellite / aerial imagery and determines, for
roofs across all of Switzerland, whether a solar installation is already present or not. The results
are linked to and cross-checked with the roof data from sonnendach.ch.

### Objectives
- Build a tool that can be applied to roofs throughout Switzerland using satellite / aerial imagery.
- Determine for each roof whether solar panels are installed.
- Link the results to the individual roof areas listed on sonnendach.ch.
- Validate the tool in two control municipalities and discuss its limitations.

### Scope and Control Municipalities
The tool is designed for the whole of Switzerland. To check the correctness of its results, two
municipalities serve as control municipalities: Uznach (Canton of St. Gallen) and Pully (Canton
of Vaud). In these two municipalities, the tool's output is verified in detail against
sonnendach.ch and the imagery, giving an assessment of how reliable the tool is.

### Data Sources
- Satellite / aerial imagery covering Switzerland (e.g. high-resolution orthophotos).
- sonnendach.ch: roof geometries and solar suitability per roof surface.
- Optionally, additional reference data for validation in the control municipalities.

### Expected Outcome
The expected result is a tool that shows, for any area in Switzerland, which roofs already carry
a solar installation and which suitable roofs are still unused. The validation in Uznach and Pully
provides an assessment of the tool's reliability. Together, this allows an estimate of the
untapped rooftop solar potential in Switzerland.

---

## 2. Our Approach in Short

| Question | Our choice | Why |
| --- | --- | --- |
| What does the model predict? | **One roof surface → PV yes / no** (image classification) | The objective is "does this roof have PV?". A classification per sonnendach roof surface can be linked directly to sonnendach (`DF_UID`). Pixel-wise segmentation would need hand-drawn masks. |
| Which model? | **ResNet18**, pretrained on ImageNet (torchvision) | Small enough to train on a laptop, well documented, easy to explain. |
| What do we train ourselves? | **Partial weights, in 3 stages** (see notebooks 05 / 06) | The goal of the module: understand CV and show how training more of the weights ourselves improves the model. |
| Which images? | **SWISSIMAGE 10 cm** (swisstopo), newest flight year | Highest free resolution in Switzerland. Even small PV plants are visible. |
| Where do the labels come from? | **We label ourselves**, one image per roof surface | The BFE register is not reliable enough on roof level (see notebook 04). |
| Training data | **Rapperswil-Jona** (SG) and **Lutry** (VD) | Neighbours of the control municipalities with the same flight year, so the images look alike. |
| Test data | **Uznach** and **Pully** | The control municipalities from the project description. Never used for training. |
| Compute | Everything **local** (Apple M4 Pro, PyTorch with MPS) | No cluster access, so the project is scaled down where needed. |

---

## 3. Notebooks

Every notebook has the same structure: **1. Goal, 2. Inputs / Outputs, 3. Analysis** (each output is followed by an
*Observation* cell with our comment), **4. Conclusion & Open Points**. Notebooks are run from top to bottom and
committed together with their outputs.

### 00 – Setup ✅
**What:** Checks Python and package versions, the GPU (Apple MPS), the config and the folder structure.
**Why:** Everything runs locally, so we make sure that the environment works before we start.

### 01 – Data Acquisition ✅
**What:** Downloads all raw data automatically: municipality boundaries (swissBOUNDARIES3D), roof surfaces from
sonnendach.ch, the BFE register of electricity production plants and the SWISSIMAGE tiles of Uznach (2025) and Pully (2023).
Cuts roofs and register entries to the two control municipalities.
**Why / decisions:**
- Only tiles that contain roofs are downloaded (Pully reaches far into Lake Geneva).
- The BFE register is used as **CSV**, because the GeoPackage version is cut off at 100'000 rows.
- Getting the raw data was complicated (different portals and formats, an archive Python can't unzip), so we used
  AI assistance (Claude) for this notebook. This is stated in the notebook.

### 02 – Data Exploration ✅
**What:** Looks at roof surfaces (size, suitability class), the register (year of commissioning) and links both via the
federal building ID (EGID). Shows single buildings on the aerial image.
**Why:** To check whether the project is feasible and how good the register is as a reference.
**Findings:**
- PV plants are clearly visible at 10 cm resolution, also small ones.
- The register is **not a reliable ground truth**: in Pully 102 of 302 plants were built after the flight,
  20–30 % of the register EGIDs are not found in sonnendach, and sometimes the EGID points to the wrong building.
- → We need our own labels.

### 03 – Roof Crops ✅
**What:** Downloads the data of the training municipalities Rapperswil-Jona and Lutry. Cuts out one image per roof
surface (everything outside the roof polygon is black) and saves a sample as PNG:
500 + 500 per training municipality, 200 + 200 per control municipality.
**Why / decisions:**
- The model should only see the one roof surface it has to classify.
- Roofs at tile borders are read from up to 4 tiles and combined.
- Only roof surfaces ≥ 10 m² (smaller ones are too small for PV).
- A random sample would contain only ~10 % PV, so half of the sample comes from buildings with a PV plant in the register
  ("register pool"), the other half from buildings without ("other pool"). Which pool an image comes from is not visible while labeling.

### 04 – Labeling and Data Split 🔄 *in progress*
**What:** A small labeling tool in the notebook: it shows one image after the other, you type `1` (PV), `0` (no PV),
`2` (unclear) or `q` (stop). Every answer is saved right away to `data/labels/labels_<municipality>.csv`, so you can
stop and continue later. Afterwards the labels are split into train / validation / test.
**Why / decisions:**
- Typing is much faster than sorting 2'800 images in the Finder, and the labels end up as CSV in git.
- Label rule: `1` as soon as modules are visible on *this* roof surface. `2` for solar thermal or when unsure.
  `0` if only the neighbour roof has modules.
- Split: Rapperswil-Jona + Lutry → train / val (80 / 20, stratified). Uznach + Pully → test. Label `2` is dropped.
**Status:**
- ✅ Lutry (1000 images: 165 PV, 808 no PV, 27 unclear)
- ✅ Pully (400 images: 56 PV, 342 no PV, 2 unclear)
- ⏳ Rapperswil-Jona (1000 images) and Uznach (400 images) still to label
**First finding:** Even though half of the images come from buildings with PV in the register, only ~15 % of the roof
surfaces really show PV. A building usually has several roof surfaces and PV is only on some of them.
→ The classes are **imbalanced** (about 1 PV : 5 no PV), which we have to handle in training.

### 05 – Model and Stage A ⏭️ *next*
**Plan:** Load the labeled images with a simple PyTorch `Dataset` (resize to 224 × 224, ImageNet normalisation,
simple augmentation: flips and 90° rotations, which is fine for top-down images).
Take ResNet18 pretrained on ImageNet, replace the last layer with a new one for 2 classes.
**Stage A:** freeze the whole network, train **only the new last layer**. This is our baseline.
Handle the class imbalance with a weighted loss. Report accuracy, precision, recall and F1 on the validation set
(accuracy alone is misleading with 85 % "no PV").

### 06 – Stages B and C (training more weights) 📋 *planned*
**Plan:**
- **Stage B:** additionally unfreeze the last ResNet block (`layer4`) and train it.
- **Stage C:** unfreeze everything and fine-tune the whole network with a small learning rate.
Compare A, B and C on the validation set: how much does it help to train more of the weights ourselves?
The best model is saved to `results/models/`.

### 07 – Evaluation in Uznach and Pully 📋 *planned*
**Plan:** Evaluate the best model on the test sets of Uznach and Pully (never seen in training).
Compare east vs. west. Look at errors (false positives, e.g. solar thermal or skylights; false negatives, e.g. dark roofs).
Use **Grad-CAM** to show which image regions the network looks at.

### 08 – Linking to sonnendach.ch and Untapped Potential 📋 *planned*
**Plan:** Run the model on **all** roof surfaces of Uznach and Pully. Link the predictions to sonnendach (`DF_UID`),
show on a map which suitable roofs (class 4–5) already have PV and which are still unused,
and estimate the untapped potential (kWh / year from sonnendach). Discuss limitations
(age of the images, solar thermal vs. PV, roof surfaces at tile borders, transfer to the rest of Switzerland).

---

## 4. Repository Structure

```
├── config.yaml          # paths, municipalities, seed
├── requirements.txt     # Python packages
├── data/
│   ├── raw/             # downloaded raw data (not in git, ~12 GB)
│   ├── interim/         # roofs and register cut to the municipalities (not in git)
│   ├── processed/crops/ # roof images per municipality (not in git)
│   └── labels/          # our labels as CSV (in git)
├── notebooks/           # 00–08, see above
├── src/                 # small shared helpers (config, seed, device)
└── results/             # figures, metrics (in git), models (not in git)
```

Raw data, crops and model weights are not in git because of their size. Run notebooks 01 and 03 to recreate them.

## 5. Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name solar-ch --display-name "Python 3 (solar-ch)"
```

Then open the notebooks with the kernel **Python 3 (solar-ch)** and run them in order.
Notebook 01 downloads about 8 GB, notebook 03 about 3.5 GB more (use WLAN, not a mobile hotspot).

## 6. Conventions

- **Simple code:** we write the code ourselves at a basic level, so that it is easy to read, debug and review.
  No complex helper classes, no heavy error handling, simple relative paths (`../data/...`).
- **AI assistance** was used for getting the raw data (notebook 01). This is marked in the notebook.
- **Seed 42** everywhere where randomness is involved.
- Coordinate system: Swiss **LV95 (EPSG:2056)** for all geodata.

## 7. Data Sources

| Dataset | Provider | Use |
| --- | --- | --- |
| [SWISSIMAGE 10 cm](https://www.swisstopo.admin.ch/de/orthobilder-swissimage-10-cm) | swisstopo | Aerial images |
| [Solar suitability of roofs (sonnendach.ch)](https://opendata.swiss/de/dataset/eignung-von-hausdachern-fur-die-nutzung-von-sonnenenergie) | BFE | Roof surfaces, suitability class, building ID |
| [Electricity production plants](https://opendata.swiss/de/dataset/elektrizitatsproduktionsanlagen) | BFE | Reference data (PV plants with building ID) |
| [swissBOUNDARIES3D](https://www.swisstopo.admin.ch/de/landschaftsmodell-swissboundaries3d) | swisstopo | Municipality boundaries |
