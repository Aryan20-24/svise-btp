# Robust plotting for Experiment 1 (pure state estimation)
# Works with the post_process_dict saved by the updated experiment_util.py

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
PNG_DIR = RESULTS_DIR / "pngs"
PP_DIR = RESULTS_DIR / "post_processing"
PNG_DIR.mkdir(parents=True, exist_ok=True)
PP_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="Plot state estimation results")
parser.add_argument("--dpath", type=str, required=True, help="Path to dataset .pt file")
args = parser.parse_args()

# -----------------------------
# Load dataset (truth + obs)
# -----------------------------
data_dict = torch.load(args.dpath)
system_choice = data_dict["name"]
t = data_dict["t"]                    # [T]
y_true = data_dict["y_true"]          # [T, d]
y_data = data_dict["y_data"]          # [T, d]
var_obs = data_dict["var"]            # [d]
d = int(data_dict["d"])
num_data = int(data_dict["num_data"])
noise_permille = int(round(float(data_dict["noise_percent"]) * 1000))
idx = int(data_dict["index"])

# -----------------------------
# Load post-processing outputs
# -----------------------------
pp_path = PP_DIR / "state_est_post_process.pt"
if not pp_path.exists():
    raise FileNotFoundError(
        f"Post-processing file not found: {pp_path}\n"
        f"Run training first, e.g.:\n"
        f"  python experiment_util.py --dpath {args.dpath}\n"
        f"so that this file is created."
    )

post = torch.load(pp_path)

# Expected by our updated experiment_util.py:
#   mu: [T_eval, d]
#   covar: [T_eval, d, d]
#   t_eval: [T_eval]
#   system: str
if not all(k in post for k in ("mu", "covar", "t_eval")):
    raise KeyError(
        "Post-processing dict does not contain required keys ('mu','covar','t_eval'). "
        "Please re-run experiment_util.py from the updated version."
    )

mu = post["mu"]            # [T_eval, d]
covar = post["covar"]      # [T_eval, d, d]
t_eval = post["t_eval"]    # [T_eval]
system_from_pp = post.get("system", system_choice)

# ±2σ envelope (diag variance)
var_diag = covar.diagonal(dim1=-2, dim2=-1)  # [T_eval, d]
sigma = var_diag.clamp_min(0).sqrt()
lb = mu - 2 * sigma
ub = mu + 2 * sigma

# -----------------------------
# Compute simple NRMSE per dim
# -----------------------------
# Interpolate mu(t_eval) to t (if lengths differ)
if len(t_eval) != len(t) or not torch.allclose(t_eval, t, atol=1e-12, rtol=0):
    # Linear interp each dimension
    mu_interp = torch.stack([
        torch.tensor(np.interp(t.numpy(), t_eval.numpy(), mu[:, j].numpy()))
        for j in range(mu.shape[-1])
    ], dim=-1)
else:
    mu_interp = mu

# NRMSE per dimension using range of true signal
eps = 1e-12
rng = (y_true.max(dim=0).values - y_true.min(dim=0).values).clamp_min(eps)
rmse = torch.sqrt(((mu_interp - y_true) ** 2).mean(dim=0))
nrmse = (rmse / rng).detach().cpu().numpy()

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(d, 1, figsize=(10, 3.2 * d), sharex=True)
axes = np.atleast_1d(axes)

for j in range(d):
    ax = axes[j]
    ax.scatter(t, y_data[:, j], s=12, color="tab:orange", alpha=0.6, label="observations")
    ax.plot(t, y_true[:, j], "k--", lw=1.2, alpha=0.8, label="true")
    ax.plot(t_eval, mu[:, j], color="tab:blue", lw=1.6, label="posterior mean")
    ax.fill_between(
        t_eval.numpy(),
        lb[:, j].numpy(),
        ub[:, j].numpy(),
        color="tab:blue",
        alpha=0.15,
        label="±2σ"
    )
    ax.set_ylabel(f"state[{j}]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

axes[-1].set_xlabel("time")
fig.suptitle(
    f"{system_from_pp} — NRMSE per dim: "
    + ", ".join([f"{v:.3f}" for v in nrmse])
    + f"  | noise={noise_permille}‰, N={num_data}, idx={idx}",
    fontsize=12
)
fig.tight_layout(rect=[0, 0.02, 1, 0.97])

# Save
system_name = system_choice.replace(" ", "_").lower()
base = f"plot_{system_name}_{noise_permille:03d}permille_{num_data:04d}_{idx:02d}"
out_path = PNG_DIR / f"{base}.png"
fig.savefig(out_path, dpi=150)
print(f"Saved plot → {out_path}")

# Also print NRMSE summary in console
for j, v in enumerate(nrmse):
    print(f"NRMSE[state {j}] = {v:.6f}")
