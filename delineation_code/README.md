# ECG delineation (ECG-Code, UNet, SegNet)

Train and evaluate **P-QRS-T delineation** on the **ISP** or **LUDB** datasets. Supported heads:

| Model (`model_params.model_name`) | Loss (`training_params.loss_class`) | Notes |
|-----------------------------------|--------------------------------------|--------|
| `ECGCode` | `ECGCodeLoss` | Interval regression head; 12-lead mel input |
| `UNetECG` | `BCEWithLogitsLoss` | 1D U-Net (`models/unet_ludb.py`) |
| `SegNetECG` | `SegnetLoss` | Bidirectional LSTM (`models/ecg_segnet.py`) |

Dense heads use `utils/unet_dataloader.py` (4-class mask: P, QRS, T, background). ECG-Code uses `utils/dataloaders.py` (`PQRSTDataloader`).

## Environment

From this directory (`delineation_code`):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Download LUDB (PhysioNet)

LUDB is the Lobachevsky University Database of electrocardiograms ([PhysioNet LUDB](XXXX)).

Mirror the release (creates `physionet.org/files/ludb/1.0.1/...` under the current directory):

```bash
mkdir -p ludb_data && cd ludb_data
wget -r -N -c -np XXXX
cd ..
```

## Run training

Run commands from **`delineation_code`**

### ECGCode — train on ISP, evaluate on ISP

```bash
cd delineation_code
python train.py \
  model_params.model_name=ECGCode \
  +model_params.init_args.n_classes=3 \
  +model_params.init_args.in_channels=12 \
  dataset=isp test_dataset=isp \
  train_csv_path=/ABS/PATH/isp_train.csv \
  test_csv_path=/ABS/PATH/isp_test.csv \
  train_data_base_path=/ABS/PATH/to/records/ \
  test_data_base_path=/ABS/PATH/to/records/ \
```

### ECGCode — train on LUDB, test on LUDB

```bash
python train.py \
  model_params.model_name=ECGCode \
  +model_params.init_args.n_classes=3 \
  +model_params.init_args.in_channels=12 \
  dataset=ludb test_dataset=ludb \
  train_csv_path=/data/ludb/ludb.csv \
  test_csv_path=/data/ludb/ludb.csv \
  train_data_base_path=/data/ludb/physionet.org/files/ludb/1.0.1/data/ \
  test_data_base_path=/data/ludb/physionet.org/files/ludb/1.0.1/data/ \
```

### UNetECG (`unet_ludb.py`)

```bash
python train.py \
  model_params.model_name=UNetECG \
  +model_params.init_args.in_channels=12 \
  +model_params.init_args.out_channels=4 \
  training_params.loss_class=BCEWithLogitsLoss \
  training_params.loss_args={} \
  dataset=ludb test_dataset=ludb \
  train_csv_path=/data/ludb/ludb_train_ids.csv \
  test_csv_path=/data/ludb/ludb_test_ids.csv \
  train_data_base_path=/data/ludb/physionet.org/files/ludb/1.0.1/data/ \
  test_data_base_path=/data/ludb/physionet.org/files/ludb/1.0.1/data/
```

### SegNetECG (`ecg_segnet.py`)

```bash
python train.py \
  model_params.model_name=SegNetECG \
  +model_params.init_args.input_size=12 \
  +model_params.init_args.hidden_size=128 \
  +model_params.init_args.num_layers=2 \
  +model_params.init_args.out_size=4 \
  training_params.loss_class=SegnetLoss \
  training_params.loss_args={} \
  dataset=isp test_dataset=isp \
  train_csv_path=/ABS/PATH/isp_train.csv \
  test_csv_path=/ABS/PATH/isp_test.csv \
  train_data_base_path=/ABS/PATH/to/records/ \
  test_data_base_path=/ABS/PATH/to/records/
```

### Evaluation only (`pred.py`)

After training, Hydra sets `save_dir` and writes `isp.pt` or `ludb.pt` under it. To run metrics on a held-out CSV:

```bash
python pred.py \
  model_params.model_name=UNetECG \
  model_params.init_args.in_channels=12 \
  model_params.init_args.out_channels=4 \
  training_params.loss_class=BCEWithLogitsLoss \
  test_dataset=ludb \
  test_csv_path=/data/ludb/ludb_test_ids.csv \
  test_data_base_path=/data/ludb/physionet.org/files/ludb/1.0.1/data/ \
  predict.checkpoint_path=/ABS/PATH/to/run/ludb.pt
```