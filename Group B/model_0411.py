"""
For each segment j of baby i (matrix n_windows x 6000):
    1D-ResNet (dilated)                 : each 30 s window -> 128 features
    attention / mean / LSTM / GRU / TF  : the windows      -> one 64-d vector
    MLP                                 : -> predicted IQ of the segment, y_ij
Then  y_i = mean of y_ij over the S segments of the baby
and   loss = (y_i - target_i)^2  (or Huber)

Evaluation: REPEATED K-fold cross-validation by baby (or by family/twin group if
a groups file is given). A baby (or a group) is never in train and test at once.
Early stopping uses validation babies taken from the training set only.

Output folder: data/model_0411_{seq_model}_{run date}, e.g. data/model_0411_lstm_20261105_1107
With --model all, every sequence model gets its own folder (same run date), plus
data/model_0411_all_{run date}/summary_models.csv with the comparison table.

Install : pip install torch numpy pandas scikit-learn scipy tqdm
Run     : python model_0411.py
          python model_0411.py --model attention --repeats 3 --folds 5
          python model_0411.py --model all --repeats 1       (compare all sequence models)
          python model_0411.py --groups-file families.csv   (columns: id,group)

All paths are relative to the folder containing this script.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
import urllib.request
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, train_test_split
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

try:
    ROOT = Path(__file__).resolve().parent
except NameError:  # notebook / interactive session
    ROOT = Path.cwd()


# =============================================================================
# CONFIG (can be overridden from the command line, see parse_args)
# =============================================================================
@dataclass
class Config:
    # ---- target ----
    target_group: str = "Note composite"                 # 1st header row
    target: str = "QIT"                                  # QIT, QIV, QIP, NCL

    # ---- data ----
    window_len: int = 6000                               # samples per window (30 s)
    sampling_rate: float = 200.0                         # Hz, only used by augmentation
    normalize_windows: bool = True                       # z-score each window
    max_windows_train: int | None = 120                  # random subsample per segment (train only)

    # ---- model ----
    seq_model: str = "lstm"                              # attention | mean | lstm | gru | transformer
    n_features_cnn: int = 128
    n_features_seq: int = 64
    base_channels: int = 16                              # must be divisible by 8
    pretrained_cnn: Path | None = None                   # .pt state_dict of a pretrained ResNet1D
    cnn_chunk: int = 60                                  # windows per CNN pass (memory)

    # ---- paths (relative to the script folder) ----
    segments_dir: Path = ROOT.parent / "data" / "segments_ecg"
    iq_file: Path = ROOT.parent / "data" / "Résultats WPPSI III.csv"
    groups_file: Path | None = None                      # optional CSV: id,group (twins/siblings)
    output_dir: Path | None = None                    
    run_stamp: str = field(default_factory=lambda: datetime.now().strftime("%m%d_%H%M"))

    # ---- training ----
    folds: int = 5
    repeats: int = 3                                     # repeated CV with different seeds
    val_fraction: float = 0.2                            # of training babies, for early stopping
    max_epochs: int = 60
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-2
    babies_per_step: int = 4                             # gradient accumulation
    loss: str = "mse"                                    # mse | huber
    augment: bool = True
    seed: int = 0

    # ---- evaluation ----
    n_bootstrap: int = 2000
    conformal_level: float = 0.90

    # ---- phone notification (ntfy) ----
    ntfy_topic: str | None = os.environ.get("NTFY_TOPIC")   # or --ntfy-topic
    ntfy_server: str = os.environ.get("NTFY_SERVER", "https://ntfy.sh")
    ntfy_test: bool = False

    def default_output_dir(self, seq_model: str | None = None) -> Path:
        name = seq_model or self.seq_model
        return ROOT.parent / "data" / f"model0411_{name}_{self.run_stamp}"

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = self.default_output_dir()


# =============================================================================
# Utils
# =============================================================================
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def autocast_ctx(device: torch.device):
    """fp16 mixed precision on NVIDIA GPUs only; plain fp32 elsewhere."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def notify(cfg: Config, title: str, message: str, tags: str = "white_check_mark",
           priority: str = "default") -> None:
    """Push a notification through ntfy. Never raises: a network problem must not kill a run.
    The title is sent in an HTTP header, so keep it ASCII."""
    if not cfg.ntfy_topic:
        return
    try:
        req = urllib.request.Request(
            f"{cfg.ntfy_server.rstrip('/')}/{cfg.ntfy_topic}", data=message.encode("utf-8"),
            headers={"Title": title, "Tags": tags, "Priority": priority}, method="POST")
        urllib.request.urlopen(req, timeout=10).close()
    except Exception as e:  # noqa: BLE001
        tqdm.write(f"(ntfy notification failed: {e})")


# =============================================================================
# Target (WPPSI-III file)
# =============================================================================
def load_iq(cfg: Config) -> dict[int, float]:
    """Read the WPPSI file (tab-separated, 2 header rows: group / column).
    Returns {baby_id: IQ}."""
    raw = None
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            raw = pd.read_csv(cfg.iq_file, sep="\t", header=None, dtype=str, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raise ValueError(f"Could not decode {cfg.iq_file}")

    groups = raw.iloc[0].ffill().fillna("").str.strip()      # merged cells -> propagate right
    names = raw.iloc[1].fillna("").str.strip()
    columns = [f"{g} - {n}" if g else n for g, n in zip(groups, names)]
    df = raw.iloc[2:].reset_index(drop=True)
    df.columns = columns

    col = f"{cfg.target_group} - {cfg.target}"
    if col not in df.columns:
        available = [c for c in columns if " - " in c and not c.endswith(" - ")]
        raise KeyError(f"Column '{col}' not found. Available: {available}")

    ids = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    iq = pd.to_numeric(df[col].str.replace(",", ".", regex=False), errors="coerce")
    ok = ids.notna() & iq.notna()
    tqdm.write(f"IQ ({col}): {ok.sum()} babies with a value, {ids.notna().sum() - ok.sum()} without")
    return dict(zip(ids[ok].astype(int), iq[ok].astype(float)))


def load_groups(cfg: Config, ids: np.ndarray) -> np.ndarray:
    """Group label per baby (siblings/twins must share a label). Default: one group per baby."""
    mapping: dict[int, str] = {}
    if cfg.groups_file is not None:
        df = pd.read_csv(cfg.groups_file)
        mapping = {int(i): str(g) for i, g in zip(df["id"], df["group"])}
    return np.array([mapping.get(int(i), f"solo_{int(i)}") for i in ids])


# =============================================================================
# Data
# =============================================================================
def resolve_path(raw: str, segments_dir: Path) -> Path:
    """index.csv may hold absolute paths, paths relative to the cwd, or just file names."""
    p = Path(raw)
    for candidate in (p, segments_dir / p, segments_dir / p.name, ROOT / p):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Segment file not found: {raw}")


class SegmentStore:
    """Loads every segment once (z-scored, float16) and serves babies to the model."""

    def __init__(self, index: pd.DataFrame, cfg: Config, device: torch.device):
        self.cfg, self.device = cfg, device
        self.paths: dict[int, list[Path]] = {
            int(i): [resolve_path(c, cfg.segments_dir) for c in chemins]
            for i, chemins in index.groupby("id")["chemin"].apply(list).items()
        }
        self.cache: dict[Path, torch.Tensor] = {}
        all_paths = [p for plist in self.paths.values() for p in plist]
        for p in tqdm(all_paths, desc="Loading segments"):   # preload: fail fast, no disk I/O per epoch
            self.cache[p] = self._load(p)

    def _load(self, path: Path) -> torch.Tensor:
        x = np.load(path).astype(np.float32)
        if x.ndim != 2 or x.shape[1] != self.cfg.window_len:
            raise ValueError(f"{path.name}: expected (n, {self.cfg.window_len}), got {x.shape}")
        if self.cfg.normalize_windows:
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
        return torch.from_numpy(x).half()

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Noise + gain jitter + slow baseline wander (realistic for ECG)."""
        n, length = x.shape
        gain = torch.empty(n, 1, device=x.device).uniform_(0.9, 1.1)
        t = torch.arange(length, device=x.device) / self.cfg.sampling_rate
        freq = torch.empty(n, 1, device=x.device).uniform_(0.1, 0.5)
        phase = torch.empty(n, 1, device=x.device).uniform_(0, 2 * math.pi)
        amp = torch.empty(n, 1, device=x.device).uniform_(0.0, 0.3)
        wander = amp * torch.sin(2 * math.pi * freq * t + phase)
        return x * gain + wander + 0.05 * torch.randn_like(x)

    def baby(self, baby_id: int, train: bool) -> list[torch.Tensor]:
        segments = []
        for p in self.paths[baby_id]:
            x = self.cache[p]
            n_max = self.cfg.max_windows_train
            if train and n_max and x.shape[0] > n_max:
                idx = torch.randperm(x.shape[0])[:n_max].sort().values   # keep temporal order
                x = x[idx]
            x = x.to(self.device).float()
            if train and self.cfg.augment:
                x = self._augment(x)
            segments.append(x)
        return segments


# =============================================================================
# Model
# =============================================================================
class ResBlock(nn.Module):
    """2 dilated convs (k=7) + GroupNorm + residual shortcut."""

    def __init__(self, c_in: int, c_out: int, stride: int, dilation: int):
        super().__init__()
        pad = 3 * dilation                                   # keeps 'same' length for k=7
        self.conv1 = nn.Conv1d(c_in, c_out, 7, stride, pad, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 7, 1, pad, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.shortcut = (nn.Identity() if c_in == c_out and stride == 1 else
                         nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False),
                                       nn.GroupNorm(8, c_out)))
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(y + self.shortcut(x))


class ResNet1D(nn.Module):
    """One 30 s window (1 x 6000) -> n_out features.

    Dilations (1, 2, 4, 8) give a receptive field of ~3000 samples (~15 s at 200 Hz)
    instead of ~3 s, so the network can see beat-to-beat variability (HRV).
    The head concatenates average AND max pooling over time."""

    def __init__(self, c: int, n_out: int):
        super().__init__()
        assert c % 8 == 0, "base_channels must be divisible by 8 (GroupNorm)"
        self.stem = nn.Sequential(nn.Conv1d(1, c, 15, 2, 7, bias=False), nn.GroupNorm(8, c),
                                  nn.ReLU(), nn.MaxPool1d(3, 2, 1))
        self.blocks = nn.Sequential(
            ResBlock(c, c, 1, 1), ResBlock(c, 2 * c, 2, 2),
            ResBlock(2 * c, 4 * c, 2, 4), ResBlock(4 * c, 8 * c, 2, 8))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(16 * c, n_out))

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1))


class SequencePooler(nn.Module):
    """Sequence of window features (1, n, d_in) -> one vector (1, d_out).

    attention   : gated attention pooling (order-free, handles any n, interpretable weights)
    mean        : mean pooling (strong, simple baseline)
    lstm / gru  : mean over all hidden states (not only the last one)
    transformer : projection + positional encoding + 2 encoder layers + mean"""

    KINDS = ("attention", "mean", "lstm", "gru", "transformer")

    def __init__(self, kind: str, d_in: int, d_out: int):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"seq_model must be one of {self.KINDS}")
        self.kind = kind
        self.norm = nn.LayerNorm(d_in)
        if kind in ("attention", "mean", "transformer"):
            self.proj = nn.Linear(d_in, d_out)
        if kind == "attention":
            self.att_v, self.att_u, self.att_w = (nn.Linear(d_out, d_out), nn.Linear(d_out, d_out),
                                                  nn.Linear(d_out, 1))
        elif kind == "lstm":
            self.rnn = nn.LSTM(d_in, d_out, batch_first=True)
        elif kind == "gru":
            self.rnn = nn.GRU(d_in, d_out, batch_first=True)
        elif kind == "transformer":
            layer = nn.TransformerEncoderLayer(d_out, nhead=4, dim_feedforward=2 * d_out,
                                               dropout=0.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)

    @staticmethod
    def positional_encoding(n: int, d: int, device) -> torch.Tensor:
        pos = torch.arange(n, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, device=device) * (-math.log(10000.0) / d))
        pe = torch.zeros(n, d, device=device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return pe

    def forward(self, x):
        x = self.norm(x)
        if self.kind == "mean":
            return self.proj(x).mean(dim=1)
        if self.kind == "attention":
            h = self.proj(x)
            a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h)))
            return (torch.softmax(a, dim=1) * h).sum(dim=1)
        if self.kind in ("lstm", "gru"):
            out, _ = self.rnn(x)
            return out.mean(dim=1)
        h = self.proj(x)
        h = h + self.positional_encoding(h.shape[1], h.shape[2], h.device)
        return self.encoder(h).mean(dim=1)


class IQModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cnn = ResNet1D(cfg.base_channels, cfg.n_features_cnn)
        self.pool = SequencePooler(cfg.seq_model, cfg.n_features_cnn, cfg.n_features_seq)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_features_seq, 32), nn.GELU(),
                                 nn.Dropout(0.3), nn.Linear(32, 1))
        self.cnn_chunk = cfg.cnn_chunk

    def encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        """(N, 6000) -> (N, n_features_cnn), in chunks, with gradient checkpointing in training."""
        x = windows.unsqueeze(1)
        feats = []
        for chunk in x.split(self.cnn_chunk):
            if self.training and torch.is_grad_enabled():
                feats.append(checkpoint(self.cnn, chunk, use_reentrant=False))
            else:
                feats.append(self.cnn(chunk))
        return torch.cat(feats)

    def forward(self, segments: list[torch.Tensor]):
        """List of S segments of one baby -> (y_i, tensor of y_ij)."""
        lengths = [s.shape[0] for s in segments]
        feats = self.encode_windows(torch.cat(segments))      # all windows in one go
        preds = torch.stack([self.mlp(self.pool(f.unsqueeze(0))).reshape(())
                             for f in feats.split(lengths)])
        return preds.mean(), preds


# =============================================================================
# Training / prediction
# =============================================================================
@torch.no_grad()
def predict(model, ids, store, mu, sd, device):
    """{id: (IQ prediction, [IQ prediction per segment])}, de-standardised."""
    model.eval()
    out = {}
    for i in ids:
        with autocast_ctx(device):
            y_hat, y_segs = model(store.baby(int(i), train=False))
        out[int(i)] = (float(y_hat) * sd + mu, (y_segs.float().cpu().numpy() * sd + mu).tolist())
    return out


def train_fold(cfg, store, ids_train, ids_val, iq, device):
    y_train = np.array([iq[int(i)] for i in ids_train])
    mu, sd = float(y_train.mean()), float(y_train.std() + 1e-8)   # train-only standardisation

    model = IQModel(cfg).to(device)
    if cfg.pretrained_cnn:
        weights = torch.load(cfg.pretrained_cnn, map_location=device, weights_only=True)
        tqdm.write(f"    pretrained CNN loaded: {model.cnn.load_state_dict(weights, strict=False)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                       patience=max(2, cfg.patience // 3))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.MSELoss() if cfg.loss == "mse" else nn.HuberLoss(delta=1.0)

    best, best_state, wait = math.inf, None, 0
    history: list[dict] = []
    n = len(ids_train)
    pbar = tqdm(range(1, cfg.max_epochs + 1), desc="  epochs", leave=False, position=1)
    for epoch in pbar:
        model.train()
        order = [int(i) for i in ids_train]
        random.shuffle(order)
        opt.zero_grad()
        total = 0.0
        for k, i in enumerate(order, start=1):
            segs = store.baby(i, train=True)
            target = torch.tensor((iq[i] - mu) / sd, device=device, dtype=torch.float32)
            with autocast_ctx(device):
                y_hat, _ = model(segs)
            loss = loss_fn(y_hat.float(), target)
            start = ((k - 1) // cfg.babies_per_step) * cfg.babies_per_step
            group_size = min(cfg.babies_per_step, n - start)   # last group can be smaller
            scaler.scale(loss / group_size).backward()
            total += loss.item()
            if k % cfg.babies_per_step == 0 or k == n:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

        preds = predict(model, ids_val, store, mu, sd, device)
        err = np.array([preds[int(i)][0] - iq[int(i)] for i in ids_val])
        val_mse = float(np.mean((err / sd) ** 2))              # same criterion as training loss
        val_mae = float(np.mean(np.abs(err)))
        sched.step(val_mse if np.isfinite(val_mse) else 1e9)
        history.append({"epoch": epoch, "train_loss": total / n, "val_mse": val_mse,
                        "val_mae": val_mae, "lr": opt.param_groups[0]["lr"]})
        pbar.set_postfix(train=f"{total / n:.3f}", val_mse=f"{val_mse:.3f}",
                         val_mae=f"{val_mae:.2f}", lr=f"{opt.param_groups[0]['lr']:.0e}")

        if np.isfinite(val_mse) and val_mse < best - 1e-4:
            best, best_state, wait = val_mse, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                tqdm.write(f"    early stop (best val MSE(z): {best:.3f})")
                break

    pbar.close()
    if best_state is None:
        tqdm.write("    WARNING: validation never improved / was NaN -> keeping last weights")
    else:
        model.load_state_dict(best_state)
    return model, mu, sd, history


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(y, p) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_tot = np.sum((y - y.mean()) ** 2)
    constant = np.std(p) < 1e-12
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "R2": float(1 - np.sum((y - p) ** 2) / ss_tot),
        "Pearson": np.nan if constant else float(pearsonr(y, p)[0]),
        "Spearman": np.nan if constant else float(spearmanr(y, p).correlation),
        "within15": float(100 * np.mean(np.abs(y - p) <= 15)),
    }


def print_metrics(name: str, m: dict[str, float]) -> None:
    tqdm.write(f"  {name:24s} MAE {m['MAE']:6.2f} | RMSE {m['RMSE']:6.2f} | R2 {m['R2']:6.2f} | "
               f"Pearson {m['Pearson']:5.2f} | Spearman {m['Spearman']:5.2f} | "
               f"within +-15: {m['within15']:5.1f} %")


def bootstrap_ci(y, p, baseline, n_boot, seed):
    """95 % CI over babies for MAE, R2 and the MAE gain versus the baseline."""
    rng = np.random.default_rng(seed)
    y, p, baseline = map(np.asarray, (y, p, baseline))
    maes, r2s, gains = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yy, pp, bb = y[idx], p[idx], baseline[idx]
        ss_tot = np.sum((yy - yy.mean()) ** 2)
        if ss_tot == 0:
            continue
        maes.append(np.mean(np.abs(yy - pp)))
        r2s.append(1 - np.sum((yy - pp) ** 2) / ss_tot)
        gains.append(np.mean(np.abs(yy - bb)) - np.mean(np.abs(yy - pp)))
    q = lambda a: np.percentile(a, [2.5, 97.5])
    return {"MAE": q(maes), "R2": q(r2s), "MAE_gain_vs_baseline": q(gains)}


# =============================================================================
# Training diagnostics (underfit vs overfit)
# =============================================================================
def fold_summary(hist: list[dict]) -> dict[str, float]:
    """train_end = mean train loss over the last 3 epochs; val_best = best validation MSE.
    Both are in z-score units, where ~1.0 means 'no better than predicting the mean'."""
    h = pd.DataFrame(hist)
    return {"train_end": float(h["train_loss"].tail(3).mean()),
            "val_best": float(h["val_mse"].min()), "epochs": int(len(h))}


def verdict(train_end: float, val_best: float) -> str:
    # heuristic thresholds; train loss is measured with dropout + augmentation ON, so it reads high
    if val_best < 0.75:
        return "SOME SIGNAL (validation MSE clearly below 1)"
    if train_end > 0.85:
        return "UNDERFIT (train loss ~1: the network is not even fitting the training babies)"
    if train_end < 0.6 and val_best > 0.8:
        return "OVERFIT (fits train, does not transfer to unseen babies)"
    return "in between (partial fit, no clear generalisation)"


def plot_history(hist_df: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        tqdm.write("(matplotlib not installed: skipping training_curves.png)")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for (_, _), g in hist_df.groupby(["repeat", "fold"]):
        ax.plot(g["epoch"], g["train_loss"], color="tab:blue", alpha=0.25)
        ax.plot(g["epoch"], g["val_mse"], color="tab:orange", alpha=0.25)
    mean = hist_df.groupby("epoch")[["train_loss", "val_mse"]].mean()
    ax.plot(mean.index, mean["train_loss"], color="tab:blue", lw=2.5, label="train loss (mean over folds)")
    ax.plot(mean.index, mean["val_mse"], color="tab:orange", lw=2.5, label="validation MSE (mean over folds)")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="~ predicting the mean")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (z-scored IQ)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main(cfg: Config) -> dict:
    device = get_device()
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "models").mkdir(exist_ok=True)
    (cfg.output_dir / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                   indent=2, ensure_ascii=False))
    tqdm.write(f"Device: {device} | sequence model: {cfg.seq_model} | output: {cfg.output_dir}")

    index = pd.read_csv(cfg.segments_dir / "index.csv")
    iq = load_iq(cfg)
    with_ecg = set(int(i) for i in index["id"].unique())
    ids = np.array(sorted(with_ecg & set(iq)))
    no_iq = sorted(with_ecg - set(iq))
    tqdm.write(f"{len(ids)} babies with ECG and IQ"
               + (f" ({len(no_iq)} ECG without IQ ignored: {no_iq})" if no_iq else ""))

    group_of = load_groups(cfg, ids)
    unique_groups = np.unique(group_of)
    n_folds = min(cfg.folds, len(unique_groups))
    if n_folds < 3:
        raise ValueError("Too few babies/groups for cross-validation.")

    store = SegmentStore(index[index["id"].isin(ids)], cfg, device)

    oof_rows, seg_rows, repeat_metrics, hist_rows, fold_stats = [], [], [], [], []
    t0 = time.time()
    fold_bar = tqdm(total=cfg.repeats * n_folds, desc="Folds", position=0)
    for rep in range(cfg.repeats):
        seed = cfg.seed + rep
        set_seed(seed)
        rep_rows = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (g_tr, g_te) in enumerate(kf.split(unique_groups), start=1):
            train_groups = unique_groups[g_tr]
            test_ids = ids[np.isin(group_of, unique_groups[g_te])]
            # validation split is also done by group, never by single baby
            tr_groups, val_groups = train_test_split(
                train_groups, test_size=max(1, round(cfg.val_fraction * len(train_groups))),
                random_state=seed)
            train_ids = ids[np.isin(group_of, tr_groups)]
            val_ids = ids[np.isin(group_of, val_groups)]
            tqdm.write(f"\nRepeat {rep + 1}/{cfg.repeats} - fold {fold}/{n_folds}: "
                       f"{len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

            model, mu, sd, hist = train_fold(cfg, store, train_ids, val_ids, iq, device)
            hist_rows += [{"repeat": rep + 1, "fold": fold, **r} for r in hist]
            pd.DataFrame(hist_rows).to_csv(cfg.output_dir / "history.csv", index=False)
            fs = fold_summary(hist)
            fold_stats.append(fs)
            tqdm.write(f"    diagnosis: train loss {fs['train_end']:.2f} | best val MSE {fs['val_best']:.2f} "
                       f"| {fs['epochs']} epochs -> {verdict(fs['train_end'], fs['val_best'])}")
            torch.save(model.state_dict(), cfg.output_dir / "models" / f"rep{rep + 1}_fold{fold}.pt")

            preds = predict(model, test_ids, store, mu, sd, device)
            for i in test_ids:
                i = int(i)
                rep_rows.append({"id": i, "repeat": rep + 1, "fold": fold, "iq_true": iq[i],
                                 "iq_pred": preds[i][0], "iq_baseline": mu})
                seg_rows += [{"id": i, "repeat": rep + 1, "fold": fold, "segment": s, "iq_pred": v}
                             for s, v in enumerate(preds[i][1])]
            fold_bar.update(1)

        rep_df = pd.DataFrame(rep_rows)
        oof_rows += rep_rows
        m = compute_metrics(rep_df["iq_true"], rep_df["iq_pred"])
        repeat_metrics.append(m)
        tqdm.write(f"\n--- Repeat {rep + 1} (out-of-fold) ---")
        print_metrics("Model", m)
        print_metrics("Baseline (train mean)", compute_metrics(rep_df["iq_true"], rep_df["iq_baseline"]))

    fold_bar.close()
    hist_df = pd.DataFrame(hist_rows)
    plot_history(hist_df, cfg.output_dir / "training_curves.png")
    train_end = float(np.mean([f["train_end"] for f in fold_stats]))
    val_best = float(np.mean([f["val_best"] for f in fold_stats]))
    tqdm.write(f"\nTraining diagnosis (mean over folds): train loss {train_end:.2f} | "
               f"best val MSE {val_best:.2f} -> {verdict(train_end, val_best)}")
    oof = pd.DataFrame(oof_rows)
    pd.DataFrame(seg_rows).to_csv(cfg.output_dir / "predictions_per_segment.csv", index=False)
    oof.to_csv(cfg.output_dir / "predictions_per_repeat.csv", index=False)

    # one row per baby: average of its out-of-fold predictions over the repeats
    final = (oof.groupby("id")
                .agg(iq_true=("iq_true", "first"), iq_pred=("iq_pred", "mean"),
                     iq_pred_std=("iq_pred", "std"), iq_baseline=("iq_baseline", "mean"))
                .reset_index())
    final.to_csv(cfg.output_dir / "predictions_dl.csv", index=False)

    tqdm.write(f"\nDone in {(time.time() - t0) / 60:.1f} min. Files in {cfg.output_dir}\n")
    if cfg.repeats > 1:
        tqdm.write("Variability across repeats (mean +- std):")
        for key in ("MAE", "RMSE", "R2", "Spearman"):
            vals = np.array([m[key] for m in repeat_metrics])
            tqdm.write(f"  {key:9s} {np.nanmean(vals):6.2f} +- {np.nanstd(vals):.2f}")

    tqdm.write("\nFinal results (cross-validation by baby, mean over repeats):")
    y, p, b = final["iq_true"].values, final["iq_pred"].values, final["iq_baseline"].values
    print_metrics(f"Model ({cfg.seq_model})", compute_metrics(y, p))
    print_metrics("Baseline (train mean)", compute_metrics(y, b))
    ci = bootstrap_ci(y, p, b, cfg.n_bootstrap, cfg.seed)
    tqdm.write(f"  95% CI  MAE [{ci['MAE'][0]:.2f}, {ci['MAE'][1]:.2f}] | "
               f"R2 [{ci['R2'][0]:.2f}, {ci['R2'][1]:.2f}] | "
               f"MAE gain vs baseline [{ci['MAE_gain_vs_baseline'][0]:.2f}, "
               f"{ci['MAE_gain_vs_baseline'][1]:.2f}]")
    if ci["MAE_gain_vs_baseline"][0] <= 0:
        tqdm.write("  -> the gain over the baseline is NOT clearly different from 0.")

    # Indicative prediction interval (split-conformal on out-of-fold errors)
    err = np.abs(y - p)
    n = len(err)
    level = min(1.0, math.ceil((n + 1) * cfg.conformal_level) / n)
    q = np.quantile(err, level, method="higher")
    tqdm.write(f"\nIndicative {int(cfg.conformal_level * 100)} % prediction interval: "
               f"predicted IQ +- {q:.1f} points (approximate: small n, errors from different models)")

    m_final, m_base = compute_metrics(y, p), compute_metrics(y, b)
    return {"model": cfg.seq_model, "MAE": m_final["MAE"], "RMSE": m_final["RMSE"],
            "R2": m_final["R2"], "Spearman": m_final["Spearman"],
            "baseline_MAE": m_base["MAE"], "MAE_gain": m_base["MAE"] - m_final["MAE"],
            "MAE_gain_CI_low": float(ci["MAE_gain_vs_baseline"][0]),
            "MAE_gain_CI_high": float(ci["MAE_gain_vs_baseline"][1]),
            "std_MAE_over_repeats": float(np.std([m["MAE"] for m in repeat_metrics])),
            "train_loss_end": train_end, "val_mse_best": val_best,
            "diagnosis": verdict(train_end, val_best).split(" (")[0]}


def parse_args() -> Config:
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", choices=SequencePooler.KINDS + ("all",), default=cfg.seq_model,
                    help="sequence model, or 'all' to run every one (~5x longer)")
    ap.add_argument("--target", default=cfg.target, help="QIT, QIV, QIP, NCL")
    ap.add_argument("--folds", type=int, default=cfg.folds)
    ap.add_argument("--repeats", type=int, default=cfg.repeats)
    ap.add_argument("--epochs", type=int, default=cfg.max_epochs)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--groups-file", type=Path, default=None, help="CSV with columns id,group")
    ap.add_argument("--pretrained-cnn", type=Path, default=None)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--ntfy-topic", default=None, help="ntfy topic (or env NTFY_TOPIC)")
    ap.add_argument("--ntfy-test", action="store_true", help="send a test notification and exit")
    a = ap.parse_args()
    cfg.seq_model, cfg.target, cfg.folds, cfg.repeats = a.model, a.target, a.folds, a.repeats
    cfg.max_epochs, cfg.seed, cfg.groups_file = a.epochs, a.seed, a.groups_file
    cfg.pretrained_cnn, cfg.augment = a.pretrained_cnn, not a.no_augment
    cfg.ntfy_topic = a.ntfy_topic or cfg.ntfy_topic
    cfg.ntfy_test = a.ntfy_test
    cfg.output_dir = cfg.default_output_dir()      # recompute now that --model is known
    return cfg


def run_all(cfg: Config) -> pd.DataFrame:
    """Run every sequence model with the SAME seeds/folds (paired comparison).
    Each model gets its own folder model_0411_{kind}_{run_stamp}."""
    rows = []
    for kind in SequencePooler.KINDS:
        tqdm.write(f"\n{'=' * 78}\nSEQUENCE MODEL: {kind}\n{'=' * 78}")
        rows.append(main(replace(cfg, seq_model=kind, output_dir=cfg.default_output_dir(kind))))
    table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    summary_dir = cfg.default_output_dir("all")
    summary_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(summary_dir / "summary_models.csv", index=False)
    tqdm.write(f"\n{'=' * 78}\nSUMMARY (sorted by MAE) -> {summary_dir / 'summary_models.csv'}\n{'=' * 78}")
    tqdm.write(table.round(3).to_string(index=False))
    tqdm.write("\nCaution: picking the best model on these same scores is optimistic (selection bias).\n"
               "Report all of them, or confirm the winner on data never used for this comparison.\n"
               "If the MAE gain CIs overlap or include 0, the models are not distinguishable.")
    return table


def summarize(res: dict, minutes: float) -> str:
    return (f"{res['model']}: MAE {res['MAE']:.2f} (baseline {res['baseline_MAE']:.2f}) | "
            f"R2 {res['R2']:.2f} | gain {res['MAE_gain']:+.2f} "
            f"| train {res['train_loss_end']:.2f} val {res['val_mse_best']:.2f} ({res['diagnosis']}) "
            f"[{res['MAE_gain_CI_low']:+.2f}, {res['MAE_gain_CI_high']:+.2f}] | {minutes:.0f} min")


if __name__ == "__main__":
    _cfg = parse_args()
    if _cfg.ntfy_test:
        if not _cfg.ntfy_topic:
            raise SystemExit("No topic: use --ntfy-topic <name> or set NTFY_TOPIC.")
        notify(_cfg, "ntfy test", "Notification from model_pipeline.py works.")
        raise SystemExit("Test notification sent (check your phone).")
    _t = time.time()
    try:
        if _cfg.seq_model == "all":
            _table = run_all(_cfg)
            _minutes = (time.time() - _t) / 60
            _msg = "\n".join(summarize(r, _minutes) for r in _table.to_dict("records"))
            notify(_cfg, f"Training done - all models ({_minutes:.0f} min)", _msg)
        else:
            _res = main(_cfg)
            notify(_cfg, f"Training done - {_cfg.seq_model}", summarize(_res, (time.time() - _t) / 60))
    except KeyboardInterrupt:
        notify(_cfg, "Training interrupted", "Stopped manually (Ctrl+C).", tags="warning")
        raise
    except Exception as _e:
        notify(_cfg, "Training FAILED", f"{type(_e).__name__}: {_e}", tags="x", priority="high")
        raise