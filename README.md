# ECG Dataset with Multi-Expert Annotations and Delineations

This repository contains the experiment code for the paper **"ECG Dataset with Multi-Expert Annotations and Delineations"**. It is organized around two tasks:

- `delineation_code/`: ECG wave delineation experiments for P, QRS, and T wave localization.
- `classification/`: ECG multi-label classification experiments trained on PTB-XL and evaluated on the ISP multi-expert dataset.

The code assumes that dataset files are prepared locally. Large raw ECG files, prepared arrays, model checkpoints, and experiment outputs are intentionally kept outside the source tree unless explicitly noted.