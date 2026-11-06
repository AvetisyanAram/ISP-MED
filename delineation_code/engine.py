import torch
from tqdm import tqdm

from metric_utils import metrics_calculation, stopping_criterion
from postprocessing import calculate_mask, remove_overlaps
import models


def get_model(config):
    model_init_args = config.model_params.init_args
    model = getattr(models, config.model_params.model_name)(**model_init_args)
    return model


def train_epoch(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    epochs_no_improve,
    best_metrics,
    device,
    cfg,
    epoch,
):
    train_loss = train_fn(train_loader, model, optimizer, criterion, device, epoch)
    val_loss = eval_fn(
        valid_loader,
        model,
        criterion,
        device,
        cfg.training_params.window_tolerance,
        cfg,
    )
    scheduler.step(val_loss)
    epochs_no_improve, best_metrics = stopping_criterion(
        val_loss, best_metrics, epochs_no_improve
    )
    if epochs_no_improve == 0:
        print("Model saved!")
        model_info = {
            "model": model.state_dict(),
            "config_file": cfg,
            "metrics": {
                "val_loss": val_loss,
            },
        }
        checkpoint_path = f"{cfg.save_dir}/{cfg.dataset}.pt"
        torch.save(model_info, checkpoint_path)

    return train_loss, val_loss, epochs_no_improve, best_metrics


def train_fn(data_loader, model, optimizer, criterion, device, epoch):
    sum_loss = 0
    model.train()

    for bi, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        input = batch

        ecg_signal = input[0].to(device)
        targets = input[1].to(device)

        optimizer.zero_grad()
        outputs = model(ecg_signal)

        loss = criterion(outputs, targets)
        loss.backward()
        sum_loss += loss.detach().item()

        optimizer.step()

    return sum_loss / len(data_loader)


def eval_fn(data_loader, model, criterion, device, window_tolerance, cfg):
    model.eval()
    sum_loss = 0
    fin_targets = []
    fin_outputs = []
    file_paths = []

    with torch.no_grad():
        for bi, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            input = batch

            ecg_signal = input[0].to(device)
            targets = input[1].to(device)
            paths = input[2]

            outputs = model(ecg_signal)

            loss = criterion(outputs, targets)
            sum_loss += loss.detach().item()

            for i in range(targets.shape[0]):
                predictions = outputs[i]
                target_result, _ = calculate_mask(targets[i], cfg)
                output_result, _ = calculate_mask(predictions, cfg)
                fin_targets.append(target_result)
                fin_outputs.append(output_result)
                file_paths.append(paths[i])

    fin_outputs = [remove_overlaps(output) for output in fin_outputs]
    df_save_path = cfg.save_dir
    results = metrics_calculation(
        fin_outputs, fin_targets, window_tolerance, file_paths, df_save_path
    )
    return sum_loss / len(data_loader)
