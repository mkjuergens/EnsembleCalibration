from dataclasses import dataclass, field
from typing import Literal, Tuple, List, Any, Dict, Callable


from src.cal_estimates import (
    ece_kde_obj,
    ece_kde_obj_lambda,
    kl_kde_obj,
    kl_kde_obj_lambda,
    mmd_kce_obj,
    mmd_kce_obj_lambda,
    skce_obj,
    skce_obj_lambda,
)
from src.losses import GeneralizedBrierLoss, GeneralizedLogLoss


@dataclass
class ExperimentConfig:
    # --- all the fields as before ---
    experiment: Literal["gp", "dirichlet"] = "gp"
    n_samples: int = 2000
    n_resamples: int = 100
    n_classes: int = 10
    n_members: int = 10
    x_bound: List[float] = field(default_factory=lambda: [0.0, 5.0])
    deg: int = 2
    bounds_p: List[List[float]] = field(
        default_factory=lambda: [[0.5, 0.7], [0.6, 0.8]]
    )
    batch_size: int = 16
    device: str = "cpu"

    optim: Literal["mlp", "COBYLA", "SLSQP"] = "mlp"
    n_epochs: int = 250
    lr: float = 1e-4
    patience: int = 100
    hidden_layers: int = 3
    hidden_dim: int = 16

    loss_type: Literal["lp", "kl", "mmd", "skce"] = "lp"
    cal_weight: float = 0.5
    bw: float = 0.01
    p: int = 2

    # filled in below:
    loss: Any = None
    obj: Callable = None
    obj_lambda: Callable = None

    deg_h1_list: List[float] = field(default_factory=lambda: [0.02, 0.1, 0.15])

    @staticmethod
    def from_dict(d: dict) -> "ExperimentConfig":
        # 1) filter only the keys that ExperimentConfig actually declares
        valid_keys = set(ExperimentConfig.__dataclass_fields__.keys())
        init_kwargs = {k: v for k, v in d.items() if k in valid_keys}

        # 2) construct the dataclass
        cfg = ExperimentConfig(**init_kwargs)

        # 3) wire up the loss / obj pointers
        if cfg.loss_type == "lp":
            cfg.loss = GeneralizedBrierLoss(
                cal_loss="lp", cal_weight=cfg.cal_weight, bw=cfg.bw, p=cfg.p
            )
            cfg.obj = ece_kde_obj
            cfg.obj_lambda = ece_kde_obj_lambda
        elif cfg.loss_type == "kl":
            cfg.loss = GeneralizedLogLoss(
                cal_loss="kl", cal_weight=cfg.cal_weight, bw=cfg.bw
            )
            cfg.obj = kl_kde_obj
            cfg.obj_lambda = kl_kde_obj_lambda
        elif cfg.loss_type == "mmd":
            cfg.loss = GeneralizedBrierLoss(
                cal_loss="mmd", cal_weight=cfg.cal_weight, bw=cfg.bw
            )
            cfg.obj = mmd_kce_obj
            cfg.obj_lambda = mmd_kce_obj_lambda
        else:  # skce
            cfg.loss = GeneralizedBrierLoss(
                cal_loss="skce", cal_weight=cfg.cal_weight, bw=cfg.bw
            )
            cfg.obj = skce_obj
            cfg.obj_lambda = skce_obj_lambda

        return cfg


def build_test_params_map(cfg: ExperimentConfig) -> Dict[str, Dict[str, Any]]:
    """
    Construct a mapping from objective name to its specific params dict,
    using the common ExperimentConfig as base.
    """
    tests = {}
    # LP
    tests["LP"] = {
        "optim": cfg.optim,
        "n_samples": cfg.n_samples,
        "n_resamples": cfg.n_resamples,
        "n_classes": cfg.n_classes,
        "n_members": cfg.n_members,
        "obj": cfg.obj,
        "obj_lambda": cfg.obj_lambda,
        "bw": cfg.bw,
        "loss": cfg.loss,
        "n_epochs": cfg.n_epochs,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "patience": cfg.patience,
        "hidden_layers": cfg.hidden_layers,
        "hidden_dim": cfg.hidden_dim,
        "x_dep": cfg.x_dep,
        "deg": cfg.deg,
        "device": cfg.device,
        "bounds_p": cfg.bounds_p,
    }
    # KL
    cfg_kl = ExperimentConfig.from_dict(
        {**vars(cfg), "loss_type": "kl", "cal_weight": 0.0, "bw": cfg.bw}
    )
    tests["KL"] = {
        **vars(cfg_kl),
        "loss": cfg_kl.loss,
        "obj": cfg_kl.obj,
        "obj_lambda": cfg_kl.obj_lambda,
    }
    # MMD
    cfg_mmd = ExperimentConfig.from_dict({**vars(cfg), "loss_type": "mmd"})
    tests["MMD"] = {
        **vars(cfg_mmd),
        "loss": cfg_mmd.loss,
        "obj": cfg_mmd.obj,
        "obj_lambda": cfg_mmd.obj_lambda,
    }
    # SKCE
    cfg_skce = ExperimentConfig.from_dict({**vars(cfg), "loss_type": "skce"})
    tests["SKCE"] = {
        **vars(cfg_skce),
        "loss": cfg_skce.loss,
        "obj": cfg_skce.obj,
        "obj_lambda": cfg_skce.obj_lambda,
    }
    return tests


# Hard-coded per-objective bandwidths
BWS = dict(LP=0.01, KL=0.1, MMD=0.1, SKCE=0.0001)


def build_objective_map(cfg: ExperimentConfig, requested: list[str]) -> dict[str, dict]:
    """
    Build {name: {'params': {...}}} only for the objectives listed in `requested`.
    Each objective gets its own fixed bandwidth (see BWS above).
    """
    base = dict(
        optim=cfg.optim,
        n_resamples=cfg.n_resamples,
        n_epochs=cfg.n_epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        patience=cfg.patience,
        hidden_layers=cfg.hidden_layers,
        hidden_dim=cfg.hidden_dim,
        device=cfg.device,
    )

    obj_map: dict[str, dict] = {}

    if "LP" in requested:
        bw = BWS["LP"]
        obj_map["LP"] = {
            "params": {
                **base,
                "bw": bw,
                "p": cfg.p,  # LP needs the power-p
                "loss": GeneralizedBrierLoss(
                    cal_loss="lp", cal_weight=cfg.cal_weight, bw=bw, p=cfg.p
                ),
                "obj": ece_kde_obj,
                "obj_lambda": ece_kde_obj_lambda,
            }
        }

    if "KL" in requested:
        bw = BWS["KL"]
        obj_map["KL"] = {
            "params": {
                **base,
                "bw": bw,
                "loss": GeneralizedLogLoss(
                    cal_loss="kl", cal_weight=cfg.cal_weight, bw=bw
                ),
                "obj": kl_kde_obj,
                "obj_lambda": kl_kde_obj_lambda,
            }
        }

    if "MMD" in requested:
        bw = BWS["MMD"]
        obj_map["MMD"] = {
            "params": {
                **base,
                "bw": bw,
                "loss": GeneralizedBrierLoss(
                    cal_loss="mmd", cal_weight=cfg.cal_weight, bw=bw
                ),
                "obj": mmd_kce_obj,
                "obj_lambda": mmd_kce_obj_lambda,
            }
        }

    if "SKCE" in requested:
        bw = BWS["SKCE"]
        obj_map["SKCE"] = {
            "params": {
                **base,
                "bw": bw,
                "loss": GeneralizedBrierLoss(
                    cal_loss="skce", cal_weight=cfg.cal_weight, bw=bw
                ),
                "obj": skce_obj,
                "obj_lambda": skce_obj_lambda,
            }
        }

    if not obj_map:
        raise ValueError("`--objectives` list was empty or invalid.")

    return obj_map
