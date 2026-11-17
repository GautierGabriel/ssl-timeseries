import torch
import torch.nn as nn
import os
import inspect

import vivml.SSL.model.encoder.encoders as encoders_module
import vivml.SSL.model.pooler.poolers as poolers_module
import vivml.SSL.model.head.heads as heads_module
import vivml.SSL.model.method.methods as methods_module
import vivml.SSL.transform.augmenters as augmenters_module
import vivml.SSL.transform.normalizers as normalizers_module
import vivml.SSL.model.loss.losses as losses_module


def get_class_from_module(class_name: str):
    """Find the classes chosen in the yaml config file."""
    for module in [encoders_module, poolers_module, heads_module, normalizers_module, methods_module, augmenters_module,
                   losses_module]:
        if hasattr(module, class_name):
            return getattr(module, class_name)
    raise ValueError(f"Class '{class_name}' not found")


def build_model_from_config(config: dict):
    model_cfg = config["model"]
    method_cfg = model_cfg["method"]

    method_type = method_cfg["type"]
    method_cls = get_class_from_module(method_type)

    pretrained_weights = method_cfg.get("pretrained_weights", None)

    loss_keys = [k for k in method_cfg.keys() if k.endswith("loss_fn")]
    kwargs_keys = [k.replace("loss_fn", "loss_kwargs") for k in loss_keys]

    exclusion_keys = {"type", "pretrained_weights"}
    exclusion_keys.update(loss_keys)
    exclusion_keys.update(kwargs_keys)

    cleaned_method_kwargs = {k: v for k, v in method_cfg.items() if k not in exclusion_keys}

    components = {}

    for loss_key in loss_keys:
        loss_name = method_cfg[loss_key]
        kwarg_key = loss_key.replace("loss_fn", "loss_kwargs")
        loss_kwargs = method_cfg.get(kwarg_key, {})

        loss_cls = get_class_from_module(loss_name)
        components[loss_key] = loss_cls(**loss_kwargs)

    norm_dict = nn.ModuleDict()

    target_channels = []

    input_features = config.get("data", {}).get("input_features", {})

    for ch_name, ch_cfg in input_features.items():
        target_channels.append(ch_name)

        if ch_cfg is not None:
            norm_cfg = ch_cfg.get("normalizers", ch_cfg.get("normalizer"))
            if norm_cfg:
                comp_cls = get_class_from_module(norm_cfg["type"])
                cfg_kwargs = {k: v for k, v in norm_cfg.items() if k != "type"}

                argspec = inspect.getfullargspec(comp_cls.__init__)

                if "channel_name" in argspec.args or argspec.varkw is not None:
                    cfg_kwargs["channel_name"] = ch_name

                norm_dict[ch_name] = comp_cls(**cfg_kwargs)

    components["target_channels"] = target_channels
    components["normalizers"] = norm_dict

    latent_dim = None

    for component_name, component_cfg in model_cfg.items():
        if component_name == "method" or not isinstance(component_cfg, dict) or "type" not in component_cfg:
            continue


        comp_cls = get_class_from_module(component_cfg["type"])
        cfg = {k: v for k, v in component_cfg.items() if k != "type"}

        if component_name != "encoder" and latent_dim is not None:
            if "input_dim" in inspect.signature(comp_cls.__init__).parameters and "input_dim" not in cfg:
                cfg["input_dim"] = latent_dim

        component_instance = comp_cls(**cfg)
        components[component_name] = component_instance

        if component_name == "encoder" and hasattr(component_instance, "output_dim"):
            latent_dim = component_instance.output_dim
        elif component_name == "encoder" and "output_dim" in component_cfg:
            latent_dim = component_cfg["output_dim"]

    method_kwargs = {**cleaned_method_kwargs, **components}
    model_method = method_cls(**method_kwargs)


    return model_method

import logging

logger = logging.getLogger(__name__)


def init_model_weights(model, config, explicit_weight_path=None):
    full_weight_path = explicit_weight_path or config.get("init_weight")

    if full_weight_path and full_weight_path not in ["kaiming", "xavier"]:
        if os.path.exists(full_weight_path):
            logger.info(f"Loading full model weights from: {full_weight_path}")

            raw_checkpoint = torch.load(full_weight_path, map_location="cpu")
            state_dict = raw_checkpoint.get("state_dict", raw_checkpoint)

            clean_state_dict = {
                key.replace("_orig_mod.", ""): value
                for key, value in state_dict.items()
            }

            model.load_state_dict(clean_state_dict, strict=False)
            return
        else:
            logger.error(f"Weight file not found: '{full_weight_path}'")



    method_cfg = config.get("model", {}).get("method", {})
    pretrained_encoder_path = method_cfg.get("pretrained_weights")
    encoder_loaded = False

    if pretrained_encoder_path and os.path.exists(pretrained_encoder_path):
        logger.info(f"Loading pretrained encoder from: {pretrained_encoder_path}")
        checkpoint = torch.load(pretrained_encoder_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        encoder_state_dict = {
            k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")
        }
        if not encoder_state_dict:
            encoder_state_dict = {k.replace("network.", ""): v for k, v in state_dict.items() if k.startswith("network.")}

        model.encoder.load_state_dict(encoder_state_dict, strict=False)
        encoder_loaded = True

    weight_init_type = config.get("init_weight")
    if weight_init_type not in ["kaiming", "xavier"]:
        weight_init_type = "kaiming"

    initialization_seed = config.get("initialization_seed")
    if initialization_seed is not None:
        torch.manual_seed(initialization_seed)

    def init_random(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if weight_init_type == "xavier":
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    for module_name in ['head', 'pooler', 'global_residual', 'domain_head']:
        if hasattr(model, module_name) and getattr(model, module_name) is not None:
            module_instance = getattr(model, module_name)

            if module_instance.__class__.__name__ == "TSPulseReconstructionHead":
                logger.info(f"Skipping random init for HuggingFace decoder: '{module_instance.__class__.__name__}'")
                continue

            logger.info(f"Initializing {module_name} with random weights ({weight_init_type}).")
            module_instance.apply(init_random)

    if not encoder_loaded and hasattr(model, 'encoder'):
        encoder_class_name = model.encoder.__class__.__name__
        foundation_encoder_classes = ["TSPulseEncoder", "MomentEncoder"]

        if encoder_class_name in foundation_encoder_classes:
            logger.info(f"Skipping random init for HuggingFace encoder: '{encoder_class_name}'")
        else:
            logger.info(f"Applying random init to custom encoder: '{encoder_class_name}'")
            model.encoder.apply(init_random)