import torch
import yaml
import logging
import json
import os
import gc
from datetime import datetime
import numpy as np
import optuna
from tqdm import tqdm
import warnings
from contextlib import nullcontext
from typing import Optional

from vivml.SSL.config_yml import parse_config
from vivml.SSL.data_streamer.builder import get_dataloaders
from vivml.SSL.data_streamer.source import DataSource
from vivml.SSL.data_streamer.wrapper import stream_wrapper
from vivml.SSL.model.builder import build_model_from_config, init_model_weights
from vivml.SSL.evaluation.visualizer import TrainingVisualizer

warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")

logging.basicConfig(level=logging.WARNING)

logging.getLogger("vivml").setLevel(logging.INFO)
logging.getLogger("__main__").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def train_model(
    config,
    save_artifacts=False,
    trial=None,
    data_source: Optional[DataSource] = None,
):
    train_streamer, val_streamer = get_dataloaders(config, logger, data_source)
    try:
        return _train_model_with_streamers(
            config, save_artifacts, trial, train_streamer, val_streamer
        )
    finally:
        try:
            train_streamer.stop()
        finally:
            val_streamer.stop()


def _train_model_with_streamers(
    config, save_artifacts, trial, train_streamer, val_streamer
):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    model_method = build_model_from_config(config)
    init_model_weights(model_method, config)
    model_method.to(device)

    opt_cfg = config.get("optimizer", {})
    opt_type = opt_cfg.get("type", "Adam")
    optimizer_class = getattr(torch.optim, opt_type)
    optimizer = optimizer_class(
        model_method.parameters(),
        lr=opt_cfg.get("lr", 0.001),
        weight_decay=opt_cfg.get("weight_decay", 0.0)
    )

    num_epochs = config.get("epochs", 10)

    scheduler_config = config.get("scheduler")
    if scheduler_config and scheduler_config.get("type") == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=scheduler_config.get("eta_min", 1e-5)
        )
    else:
        scheduler = None

    eval_modulo = config.get("evaluation", {}).get("history_modulo", 10)
    checkpoint_modulo = config.get("evaluation", {}).get("checkpoint_modulo", 50)
    accumulation_steps = config.get("accumulation_steps", 1)
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be a positive integer.")
    clip_norm = config.get("optimizer", {}).get("clip_norm", 5)
    autocast_enabled = device.type == "cuda" and torch.cuda.is_bf16_supported()

    history = {"train_loss": [], "val_loss": [], "lr": []}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"{timestamp}_train_{config['experiment_name']}"
    output_dir = os.path.join("outputs", folder_name)
    history_dir = os.path.join(output_dir, "history")
    json_history_path = os.path.join(output_dir, "loss_history.json")

    if save_artifacts:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(history_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config_run.yaml"), "w") as f:
            yaml.dump(config, f, sort_keys=False)

    def dump_history():
        if not save_artifacts:
            return
        tmp_path = json_history_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(history, f, indent=4)
        os.replace(tmp_path, json_history_path)

    avg_val_loss = float('inf')

    for epoch in range(num_epochs):
        model_method.on_epoch_start(epoch, num_epochs)
        is_last_epoch = (epoch == num_epochs - 1)

        train_loader = stream_wrapper(train_streamer, device)
        model_method.train()
        train_loss_total = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{num_epochs}", leave=True)
        optimizer.zero_grad()
        pending_accumulation = 0

        for i, batch_data in enumerate(pbar):
            if hasattr(val_streamer, "queue"):
                current_queue_size = val_streamer.queue.qsize()
                max_size = config["data"].get("max_queue_size", 100)
                pbar.set_postfix({
                    "queue_fill": f"{current_queue_size}/{max_size}",
                    "status": "CPU_SLOW" if current_queue_size == 0 else "OK"
                })

            if not batch_data or len(batch_data) == 0:
                continue

            if len(batch_data) == 3:
                x_raw, y_raw, meta = batch_data
            else:
                x_raw, meta = batch_data[0], batch_data[1]
                y_raw = None

            x = x_raw.to(device) if isinstance(x_raw, torch.Tensor) else torch.tensor(x_raw, dtype=torch.float32).to(device)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            if torch.isnan(x).any() or torch.isinf(x).any():
                continue

            y = None
            if y_raw is not None:
                y = y_raw.to(device) if isinstance(y_raw, torch.Tensor) else torch.tensor(y_raw).to(device)

            if meta is None:
                meta = {"extra_channels": {}, "stds": {}}

            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if autocast_enabled
                else nullcontext()
            )
            with autocast_context:
                loss = model_method(batch_x=x, labels=y, stats=meta)

            if not torch.isfinite(loss).all():
                optimizer.zero_grad()
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}, batch {i + 1}.")

            current_loss = loss.item()

            if epoch == 0 and train_steps == 0:
                logger.info(f"Initial training loss: {current_loss:.4f}")
                dump_history()

            loss = loss / accumulation_steps
            loss.backward()
            pending_accumulation += 1

            if i == 0:
                mean_grad = 0.0
                grads = [p.grad for p in model_method.parameters() if p.grad is not None]
                if grads:
                    mean_grad = sum(g.abs().mean().item() for g in grads) / len(grads)
                logger.info(f"Epoch {epoch} | Mean Gradient={mean_grad:.8f}")

            if pending_accumulation == accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model_method.parameters(), max_norm=clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                pending_accumulation = 0

            train_loss_total += current_loss
            train_steps += 1

        if train_steps == 0:
            raise RuntimeError(f"No valid training batches were produced at epoch {epoch + 1}.")

        if pending_accumulation:
            correction = accumulation_steps / pending_accumulation
            for parameter in model_method.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(model_method.parameters(), max_norm=clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        avg_train_loss = train_loss_total / max(1, train_steps)

        train_streamer.stop()
        val_loader = stream_wrapper(val_streamer, device)

        val_loss_total, val_steps = 0.0, 0
        all_outputs, all_labels, all_metadata = [], [], []
        is_eval_epoch = is_last_epoch or (eval_modulo is not None and epoch % eval_modulo == 0)

        MAX_SAVED_SAMPLES = 2000
        current_saved_samples = 0

        model_method.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if not batch or len(batch) == 0:
                    continue
                if len(batch) == 3:
                    x_raw, y_raw, meta = batch
                else:
                    x_raw, meta = batch[0], batch[1]
                    y_raw = None

                x_val = x_raw.to(device) if isinstance(x_raw, torch.Tensor) else torch.tensor(x_raw, dtype=torch.float32).to(device)
                x_val = x_val.unsqueeze(1) if x_val.dim() == 2 else x_val
                if torch.isnan(x_val).any() or torch.isinf(x_val).any():
                    continue
                if meta is None:
                    meta = {"extra_channels": {}, "stds": {}}

                loss = model_method(batch_x=x_val, labels=y_raw, stats=meta)
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(
                        f"Non-finite validation loss at epoch {epoch + 1}, batch {i + 1}."
                    )

                if val_steps == 0:
                    for k, v in model_method.compute_training_metrics(batch_x=x_val, stats=meta).items():
                        history.setdefault(k, []).append(v)

                val_loss_total += loss.item()
                val_steps += 1

                if save_artifacts and is_eval_epoch and current_saved_samples < MAX_SAVED_SAMPLES:
                    proj = model_method.get_projection(x_val, stats=meta).detach().cpu().numpy()
                    all_outputs.append(proj)
                    current_saved_samples += proj.shape[0]
                    if y_raw is not None:
                        all_labels.extend(y_raw.detach().cpu().numpy().tolist())
                    if meta is not None:
                        clean_meta = {}
                        for k, v in meta.items():
                            if isinstance(v, torch.Tensor):
                                clean_meta[k] = v.detach().cpu().tolist()
                            elif isinstance(v, dict):
                                clean_meta[k] = {
                                    dk: dv.detach().cpu().tolist() if isinstance(dv, torch.Tensor) else dv
                                    for dk, dv in v.items()
                                }
                            else:
                                clean_meta[k] = v
                        all_metadata.append(clean_meta)

        avg_val_loss = val_loss_total / max(1, val_steps)
        val_streamer.stop()
        if val_steps == 0:
            raise RuntimeError(f"No valid validation batches were produced at epoch {epoch + 1}.")

        if trial is not None:
            trial.report(avg_val_loss, step=epoch)
            if trial.should_prune():
                train_streamer.stop()
                val_streamer.stop()
                raise optuna.TrialPruned()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        logger.info(
            f"Epoch {epoch + 1:03d}/{num_epochs:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}"
        )

        dump_history()

        if save_artifacts and is_eval_epoch:
            npz_path = os.path.join(history_dir, f"epoch_{epoch:03d}_latent.npz")
            np.savez_compressed(
                npz_path,
                outputs=np.vstack(all_outputs),
                labels=np.array(all_labels) if all_labels else None,
                metadata=all_metadata if all_metadata else None
            )
            logger.info(f"Saved latent space checkpoint for epoch {epoch}")
            del all_outputs, all_labels, all_metadata
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if save_artifacts and checkpoint_modulo and epoch > 0 and epoch % checkpoint_modulo == 0 and not is_last_epoch:
            checkpoint_path = os.path.join(history_dir, f"checkpoint_epoch_{epoch:03d}.pt")
            torch.save({"epoch": epoch, "state_dict": model_method.state_dict()}, checkpoint_path)

            encoder_ckpt_path = os.path.join(history_dir, f"encoder_epoch_{epoch:03d}.pt")
            torch.save(model_method.encoder.state_dict(), encoder_ckpt_path)
            logger.info(f"Saved safety checkpoint at {checkpoint_path}")

    if save_artifacts:
        torch.save(model_method.state_dict(), os.path.join(output_dir, "full_pretext_model.pt"))
        torch.save(model_method.encoder.state_dict(), os.path.join(output_dir, "encoder_backbone.pt"))

        base_out_dir = "outputs"
        latest_model_link = os.path.join(base_out_dir, "latest_full_model.pt")
        latest_encoder_link = os.path.join(base_out_dir, "latest_encoder.pt")
        try:
            if os.path.lexists(latest_model_link):
                os.remove(latest_model_link)
            if os.path.lexists(latest_encoder_link):
                os.remove(latest_encoder_link)
            os.symlink(os.path.join(folder_name, "full_pretext_model.pt"), latest_model_link)
            os.symlink(os.path.join(folder_name, "encoder_backbone.pt"), latest_encoder_link)
            logger.info("Symlinks updated to latest models.")
        except Exception as e:
            logger.warning(f"Failed to create symlink: {e}")

        try:
            TrainingVisualizer.save_loss_curve(json_history_path, output_dir)
        except Exception as e:
            logger.error(f"Visualizer execution failed: {e}")

    return model_method, avg_val_loss, output_dir


def main(config: dict, data_source: Optional[DataSource] = None):
    logger.info(f"Start standard training: {config['experiment_name']}")
    model_method, final_loss, output_dir = train_model(
        config, save_artifacts=True, data_source=data_source
    )
    logger.info(f"Training run completed. Final Validation Loss: {final_loss:.4f}")
    print(f"RUN_DIR:{output_dir}")


if __name__ == "__main__":
    config_dict = parse_config()
    main(config_dict)
