import re
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from common import MODEL_SIZE, TASK_SET


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
    """
    Parses a Hydra config safely:
    - Converts Hydra placeholders (???)
    - Handles booleans, ints with underscores, floats, algebraic expressions
    - Processes nested dicts (e.g., distillation)
    - Respects user-provided multitask/tasks fields
    """

    # ---
    # Convert Hydra placeholders (???) -> None
    for k in list(cfg.keys()):
        try:
            if str(cfg[k]) == "???":
                cfg[k] = None
        except Exception:
            pass

    # ---
    # Booleans: None -> True, "false"/"true" strings -> bool
    for k in list(cfg.keys()):
        try:
            v = cfg[k]
            if v is None:
                cfg[k] = True
            elif isinstance(v, str) and v.lower() == "false":
                cfg[k] = False
            elif isinstance(v, str) and v.lower() == "true":
                cfg[k] = True
        except Exception:
            pass

    # ---
    # Numbers and algebraic expressions
    for k in list(cfg.keys()):
        try:
            v = cfg[k]
            if isinstance(v, str):
                # Numbers with underscores like 100_000
                if re.match(r"^\d+(_\d+)+$", v):
                    cfg[k] = int(v.replace("_", ""))
                # Simple algebraic expressions like "200*2"
                elif re.match(r"^\d+([+\-*/])\d+$", v):
                    cfg[k] = eval(v)
                    if isinstance(cfg[k], float) and cfg[k].is_integer():
                        cfg[k] = int(cfg[k])
                # Floats in scientific notation like "3e-4"
                elif re.match(r"^\d+(\.\d+)?e[+\-]?\d+$", v, re.IGNORECASE):
                    cfg[k] = float(v)
        except Exception:
            pass

    # ---
    # Handle nested "distillation" key if present
    if "distillation" in cfg and cfg.distillation is not None:
        for k, v in cfg.distillation.items():
            if isinstance(v, str):
                if re.match(r"^\d+(_\d+)+$", v):
                    cfg.distillation[k] = int(v.replace("_", ""))
                elif re.match(r"^\d+([+\-*/])\d+$", v):
                    cfg.distillation[k] = eval(v)

    # ---
    # Convenience keys
    if cfg.get("task") is not None:
        cfg.work_dir = Path(hydra.utils.get_original_cwd()) / "logs" / cfg.task / str(cfg.seed) / cfg.exp_name
        cfg.task_title = cfg.task.replace("-", " ").title()
    else:
        cfg.work_dir = Path(hydra.utils.get_original_cwd()) / "logs" / "default"
        cfg.task_title = "Unknown"

    if cfg.get("num_bins") and cfg.get("vmax") is not None and cfg.get("vmin") is not None:
        cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)

    # ---
    # Model size defaults
    if cfg.get("model_size", None) is not None:
        assert cfg.model_size in MODEL_SIZE.keys(), \
            f"Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}"
        for k, v in MODEL_SIZE[cfg.model_size].items():
            cfg[k] = v
        if cfg.task == "mt30" and cfg.model_size == 19:
            cfg.latent_dim = 512  # special case

    # ---
    # Multi-task logic (respect YAML if provided)
    if "multitask" not in cfg or cfg.multitask is None:
        cfg.multitask = cfg.task in TASK_SET.keys() if cfg.get("task") else False

    if cfg.multitask:
        cfg.task_title = cfg.task.upper()
        cfg.task_dim = 96 if (cfg.task == "mt80" or cfg.model_size in {1, 317}) else 64
    else:
        cfg.task_dim = 0

    if "tasks" not in cfg or cfg.tasks in (None, [], ["???"]):
        cfg.tasks = TASK_SET.get(cfg.task, [cfg.task]) if cfg.get("task") else []

    # ---
    # obs key normalization
    if "obs" in cfg and cfg.obs:
        if isinstance(cfg.obs, str):
            cfg.obs = [cfg.obs]
    else:
        cfg.obs = []

    return cfg
