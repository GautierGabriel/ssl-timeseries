import os
import logging
import tempfile
import torch
import numpy as np
import h5py
import gc

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseEvaluationEngine(ABC):
    def __init__(self, device: torch.device, **kwargs):
        self.device = device
        self.run_dir = kwargs.get('run_dir', tempfile.gettempdir())
        self.max_ram_samples = kwargs.get('max_ram_samples', 1000000)
    def _forward_pass(self, model, x, meta):
        return model.get_projection(x, stats=meta)

    @abstractmethod
    def compute_metrics(self, raw_results: dict) -> dict:
        pass

    def run_inference(self, model, loader) -> dict:
        model.eval()

        use_disk = False
        h5_path = os.path.join(self.run_dir, "inference_cache.h5")
        h5f, ds_embeds, ds_preds = None, None, None

        all_outputs = []
        all_prediction = []
        all_labels = []
        all_metadata = []

        current_idx = 0

        with torch.no_grad():
            for batch in loader:
                if current_idx % 5000 == 0:
                    gc.collect()
                if not batch or len(batch) == 0:
                    continue

                if len(batch) == 3:
                    x_raw, y_raw, meta = batch
                else:
                    x_raw, meta = batch[0], batch[1]
                    y_raw = None

                x = x_raw.to(self.device)
                batch_size = x.size(0)

                outputs = self._forward_pass(model, x, meta).cpu().numpy().astype(np.float16)

                if outputs.ndim == 4:
                    outputs = outputs.mean(axis=(1, 2))
                elif outputs.ndim == 3:
                    outputs = outputs.mean(axis=1)

                prediction = None
                if hasattr(model, "get_prediction"):
                    prediction = model.get_prediction(x, stats=meta).cpu().numpy().astype(np.float16)

                if not use_disk and (current_idx + batch_size) > self.max_ram_samples:
                    logger.info(f"Seuil de {self.max_ram_samples} fenetres atteint. Bascule vers le cache HDF5.")
                    use_disk = True
                    h5f = h5py.File(h5_path, 'w')

                    ds_embeds = h5f.create_dataset("embeddings", shape=(0, *outputs.shape[1:]),
                                                   maxshape=(None, *outputs.shape[1:]), dtype=outputs.dtype)

                    if prediction is not None or all_prediction:
                        p_shape = prediction.shape[1:] if prediction is not None else all_prediction[0].shape[1:]
                        p_dtype = prediction.dtype if prediction is not None else all_prediction[0].dtype
                        ds_preds = h5f.create_dataset("predictions", shape=(0, *p_shape), maxshape=(None, *p_shape),
                                                      dtype=p_dtype)

                    if current_idx > 0:
                        ds_embeds.resize(current_idx, axis=0)
                        ds_embeds[0:current_idx] = np.concatenate(all_outputs, axis=0)
                        if ds_preds is not None and all_prediction:
                            ds_preds.resize(current_idx, axis=0)
                            ds_preds[0:current_idx] = np.concatenate(all_prediction, axis=0)

                    all_outputs.clear()
                    all_prediction.clear()

                if use_disk:
                    ds_embeds.resize(current_idx + batch_size, axis=0)
                    ds_embeds[current_idx:current_idx + batch_size] = outputs

                    if prediction is not None:
                        ds_preds.resize(current_idx + batch_size, axis=0)
                        ds_preds[current_idx:current_idx + batch_size] = prediction
                else:
                    all_outputs.append(outputs)
                    if prediction is not None:
                        all_prediction.append(prediction)

                if y_raw is not None:
                    all_labels.extend(y_raw.cpu().numpy())

                if meta is not None:
                    keys_to_keep = [k for k in meta.keys() if k not in ["extra_channels", "stds", "drift"]]
                    for i in range(batch_size):
                        entry = {}
                        for k in keys_to_keep:
                            v = meta[k]
                            if isinstance(v, torch.Tensor):
                                entry[k] = v[i].item() if v.dim() == 1 else v[i].tolist()
                            elif isinstance(v, (list, tuple)) and len(v) == batch_size:
                                entry[k] = v[i]
                            else:
                                entry[k] = v
                        all_metadata.append(entry)

                current_idx += batch_size
                del x, x_raw, outputs

            if use_disk:
                logger.info("Retour des donnees via cache HDF5.")
                return {
                    "embeddings": ds_embeds,
                    "predictions": ds_preds,
                    "labels": np.array(all_labels) if all_labels else None,
                    "metadata": all_metadata if all_metadata else None,
                    "raw_windows": None,
                    "h5_file": h5f
                }
            else:
                logger.info(f"Streaming termine. {len(all_outputs)} batchs en RAM. Demarrage de la consolidation...")

                logger.info("ETAPE A : np.concatenate des embeddings...")
                concat_outputs = np.concatenate(all_outputs, axis=0) if all_outputs else np.array([])
                logger.info(
                    f"ETAPE A OK : Shape final = {concat_outputs.shape}, Poids estimé = {concat_outputs.nbytes / (1024 ** 3):.2f} Go")
                del all_outputs  # Liberation immediate de la liste Python

                logger.info("ETAPE B : np.concatenate des predictions...")
                concat_preds = np.concatenate(all_prediction, axis=0) if all_prediction else None
                del all_prediction

                logger.info("ETAPE C : Consolidation des labels...")
                concat_labels = np.array(all_labels) if all_labels else None
                del all_labels

                logger.info("ETAPE D : Retour du dictionnaire (fin de run_inference)")
                return {
                    "embeddings": concat_outputs,
                    "predictions": concat_preds,
                    "labels": concat_labels,
                    "metadata": all_metadata,
                    "raw_windows": None
                }