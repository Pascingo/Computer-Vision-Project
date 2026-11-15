# Data

Raw data is not versioned. It is downloaded by `notebooks/01_data_acquisition.ipynb`
into `data/raw/`. Sources and download steps are documented there.

| Folder | Content | Versioned |
| --- | --- | --- |
| `raw/` | Downloaded source data (imagery, sonnendach.ch roofs, boundaries, …) | no |
| `interim/` | Intermediate products (clipped / resampled rasters) | no |
| `processed/` | Model-ready tiles, masks and splits | no |
| `labels/` | Our own annotations for the control municipalities | yes |
