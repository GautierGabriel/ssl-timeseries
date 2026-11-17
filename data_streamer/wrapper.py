import torch

def stream_wrapper(streamer, device: torch.device):
    for batch in streamer:
        x = torch.tensor(batch[0], dtype=torch.float32).to(device)
        metadata = batch[1]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if "extra_channels" in metadata:
            for ch, arr in metadata["extra_channels"].items():
                metadata["extra_channels"][ch] = torch.tensor(arr, dtype=torch.float32).to(device)

        if "stds" in metadata:
            for ch, arr in metadata["stds"].items():
                metadata["stds"][ch] = torch.tensor(arr, dtype=torch.float32).to(device)

        if "drift" in metadata:
            metadata["drift"] = torch.tensor(metadata["drift"], dtype=torch.float32).to(device)

        if "start_idx" in metadata:
            metadata["start_idx"] = torch.tensor(metadata["start_idx"], dtype=torch.long).to(device)

        has_labels = "label" in metadata and len(metadata["label"]) > 0 and metadata["label"][0] is not None

        if has_labels:
            labels_tensor = torch.tensor(metadata["label"], dtype=torch.long).to(device)
            yield (x, labels_tensor, metadata)
        else:
            yield (x, None, metadata)