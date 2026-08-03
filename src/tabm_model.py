"""TabM (parameter-efficient MLP ensemble, Gorishniy et al., ICLR 2025) for
the freight-rate features.

TabM trains k weight-sharing MLPs in parallel and averages their predictions,
giving deep-ensemble accuracy at roughly single-MLP cost. It is the strongest
recent *trained* neural baseline on tabular benchmarks (TabArena / TALENT),
and — unlike the TabPFN/TabICL foundation models — needs no pretrained
checkpoint, so it runs in any environment.

Protocol matches the LightGBM reference: fit on log(posted_rate), early-stop
on the same held-out frame, deterministic under a fixed seed. Numeric
features are z-scored and passed through piecewise-linear embeddings (the
paper's recommended variant); categoricals use the aligned pandas category
codes (align_categories must have been applied to every frame involved).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

BATCH_SIZE = 256
LR = 2e-3
WEIGHT_DECAY = 3e-4
MAX_EPOCHS = 400
PATIENCE = 16


class TabMModel:
    """Fitted TabM with the preprocessing statistics baked in."""

    def __init__(self, net, num_mean, num_std, y_mean, y_std):
        self.net = net
        self.num_mean, self.num_std = num_mean, num_std
        self.y_mean, self.y_std = y_mean, y_std

    def _tensors(self, df: pd.DataFrame):
        x_num = (df[NUMERIC_FEATURES].to_numpy(np.float32) - self.num_mean) / self.num_std
        x_cat = np.stack(
            [df[c].cat.codes.to_numpy() for c in CATEGORICAL_FEATURES], axis=1
        ).astype(np.int64)
        return torch.from_numpy(x_num), torch.from_numpy(x_cat)

    @torch.no_grad()
    def predict_log(self, df: pd.DataFrame, batch: int = 4096) -> np.ndarray:
        self.net.eval()
        x_num, x_cat = self._tensors(df)
        out = []
        for i in range(0, len(x_num), batch):
            pred = self.net(x_num[i : i + batch], x_cat[i : i + batch])
            out.append(pred.squeeze(-1).mean(dim=1))  # average the k submodels
        return (torch.cat(out).numpy() * self.y_std) + self.y_mean


def train_tabm(train_f: pd.DataFrame, stop_f: pd.DataFrame, seed: int = 42) -> TabMModel:
    import rtdl_num_embeddings
    from tabm import TabM

    torch.manual_seed(seed)
    torch.set_num_threads(4)

    num_mean = train_f[NUMERIC_FEATURES].to_numpy(np.float32).mean(axis=0)
    num_std = train_f[NUMERIC_FEATURES].to_numpy(np.float32).std(axis=0)
    num_std[num_std < 1e-6] = 1.0
    y_log = np.log(train_f["posted_rate"].to_numpy(np.float32))
    y_mean, y_std = float(y_log.mean()), float(y_log.std())

    cardinalities = [len(train_f[c].cat.categories) for c in CATEGORICAL_FEATURES]
    x_num = torch.from_numpy(
        (train_f[NUMERIC_FEATURES].to_numpy(np.float32) - num_mean) / num_std
    )
    x_cat = torch.from_numpy(
        np.stack(
            [train_f[c].cat.codes.to_numpy() for c in CATEGORICAL_FEATURES], axis=1
        ).astype(np.int64)
    )
    y = torch.from_numpy((y_log - y_mean) / y_std)

    # Piecewise-linear embeddings are the paper's recommended numeric encoding;
    # binary flag columns can make bin computation degenerate, so fall back to
    # a plain TabM if that happens.
    try:
        bins = rtdl_num_embeddings.compute_bins(x_num)
        embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            bins, 16, activation=False, version="B"
        )
    except Exception:
        embeddings = None
    net = TabM.make(
        n_num_features=len(NUMERIC_FEATURES),
        cat_cardinalities=cardinalities,
        d_out=1,
        num_embeddings=embeddings,
    )
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    model = TabMModel(net, num_mean, num_std, y_mean, y_std)
    stop_y = stop_f["posted_rate"].to_numpy()
    gen = torch.Generator().manual_seed(seed)
    best_mae, best_state, bad_epochs = np.inf, None, 0

    for epoch in range(MAX_EPOCHS):
        net.train()
        for idx in torch.randperm(len(y), generator=gen).split(BATCH_SIZE):
            opt.zero_grad()
            out = net(x_num[idx], x_cat[idx]).squeeze(-1)  # (batch, k)
            loss = F.mse_loss(out, y[idx].unsqueeze(1).expand_as(out))
            loss.backward()
            opt.step()

        stop_pred = np.exp(model.predict_log(stop_f))
        mae = float(np.mean(np.abs(stop_pred - stop_y)))
        if mae < best_mae - 1e-3:
            best_mae, bad_epochs = mae, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
        if epoch % 10 == 0:
            print(f"tabm epoch {epoch}: stop MAE ${mae:.2f} (best ${best_mae:.2f})", flush=True)

    net.load_state_dict(best_state)
    print(f"tabm done: best stop MAE ${best_mae:.2f} after {epoch + 1} epochs", flush=True)
    return model
