import numpy as np
import torch
import pickle
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import energyflow as ef

# ── Constants ──────────────────────────────────────────────────────────────────
K_NEIGHBORS   = 16
MAX_PARTICLES = 150
N_PER_CLASS   = 50_000   # 50k quarks + 50k gluons = 100k total
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15

PDG_MAP = {
    22:0,
    211:1,  -211:1,
    321:2,  -321:2,
    2212:3, -2212:3,
    130:4,
    11:5,   -11:5,
    13:6,   -13:6,
    2112:7, -2112:7,
}
N_PDG = len(set(PDG_MAP.values())) + 1

def encode_pdg(p):
    return PDG_MAP.get(int(p), N_PDG - 1)

# ── Graph builder ──────────────────────────────────────────────────────────────
def jet_to_graph(particles):
    mask      = particles[:, 0] > 0
    particles = particles[mask][:MAX_PARTICLES]
    if len(particles) == 0:
        return None
    n = len(particles)

    pt    = particles[:, 0]
    eta   = particles[:, 1]
    phi   = particles[:, 2]
    pdgid = particles[:, 3]

    jet_pt   = pt.sum() + 1e-8
    E_approx = pt * np.cosh(eta)
    jet_eta  = np.sum(pt * eta) / jet_pt
    jet_phi  = np.sum(pt * phi) / jet_pt

    pt_rel  = pt / jet_pt
    eta_rel = eta - jet_eta
    phi_rel = phi - jet_phi
    phi_rel = (phi_rel + np.pi) % (2 * np.pi) - np.pi
    log_pt  = np.log(pt + 1e-8)
    log_E   = np.log(E_approx + 1e-8)
    pdg_enc = np.array([encode_pdg(p) / N_PDG for p in pdgid])

    x = torch.tensor(
        np.stack([pt_rel, eta_rel, phi_rel,
                  log_pt, log_E, pdg_enc], axis=1).astype(np.float32)
    )

    coords     = torch.tensor(
        np.stack([eta_rel, phi_rel], axis=1).astype(np.float32)
    )
    k          = min(K_NEIGHBORS, n - 1)
    edge_index = knn_graph(coords, k=k, loop=False)

    src, dst = edge_index
    d_eta  = eta_rel[src.numpy()] - eta_rel[dst.numpy()]
    d_phi  = phi_rel[src.numpy()] - phi_rel[dst.numpy()]
    d_phi  = (d_phi + np.pi) % (2 * np.pi) - np.pi
    dR     = np.sqrt(d_eta**2 + d_phi**2 + 1e-8)
    kt     = np.minimum(pt[src.numpy()], pt[dst.numpy()]) * dR
    log_kt = np.log(kt + 1e-8)

    edge_attr = torch.tensor(
        np.stack([d_eta, d_phi, dR, log_kt], axis=1).astype(np.float32)
    )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ── Dataset loader ─────────────────────────────────────────────────────────────
def load_dataset(n_per_class=N_PER_CLASS, seed=42,
                 graphs_cache_dir=None, ef_cache_dir=None):

    tag      = f"balanced_n{n_per_class}_s{seed}"
    ef_cache = str(ef_cache_dir) if ef_cache_dir else "~/.energyflow"

    # ── Load from graph cache if exists ───────────────────────────────────────
    if graphs_cache_dir is not None:
        cache_file = Path(graphs_cache_dir) / f"{tag}.pkl"
        if cache_file.exists():
            print(f"[data_loader] Loading from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                train, val, test = pickle.load(f)
            print(f"[data_loader] train={len(train):,}  "
                  f"val={len(val):,}  test={len(test):,}")
            return train, val, test

    # ── Download from EnergyFlow ───────────────────────────────────────────────
    print(f"[data_loader] Downloading dataset ...")
    print(f"[data_loader] Loading 100,000 jets (pythia, no b/c)")

    X, y = ef.qg_jets.load(
        num_data  = 100_000,
        pad       = True,
        generator = 'pythia',
        with_bc   = False,
        cache_dir = ef_cache,
    )

    print(f"[data_loader] Raw dataset shape : X={X.shape}  y={y.shape}")
    print(f"[data_loader] Quarks in raw     : {(y==1).sum():,}")
    print(f"[data_loader] Gluons in raw     : {(y==0).sum():,}")

    # ── Balanced sampling ─────────────────────────────────────────────────────
    q_idx = np.where(y == 1)[0][:n_per_class]
    g_idx = np.where(y == 0)[0][:n_per_class]

    X_q = X[q_idx]
    X_g = X[g_idx]

    print(f"[data_loader] Sampled quarks    : {len(X_q):,}")
    print(f"[data_loader] Sampled gluons    : {len(X_g):,}")

    X_all = np.concatenate([X_q, X_g], axis=0)
    y_all = np.array(
        [1] * len(X_q) + [0] * len(X_g),
        dtype=np.int64
    )

    # ── Shuffle ───────────────────────────────────────────────────────────────
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(len(y_all))
    X_all = X_all[perm]
    y_all = y_all[perm]

    # ── Build graphs ──────────────────────────────────────────────────────────
    print(f"[data_loader] Building graphs from {len(y_all):,} jets ...")
    graphs = []
    for i, (particles, label) in enumerate(zip(X_all, y_all)):
        if i % 10_000 == 0:
            print(f"  {i:>7,} / {len(y_all):,}", end="\r")
        g = jet_to_graph(particles)
        if g is not None:
            g.y = torch.tensor([int(label)], dtype=torch.long)
            graphs.append(g)

    print(f"\n[data_loader] Valid graphs      : {len(graphs):,}")

    # ── Split ─────────────────────────────────────────────────────────────────
    n_train = int(TRAIN_FRAC * len(graphs))
    n_val   = int(VAL_FRAC   * len(graphs))
    train   = graphs[:n_train]
    val     = graphs[n_train : n_train + n_val]
    test    = graphs[n_train + n_val:]

    print(f"[data_loader] train={len(train):,}  "
          f"val={len(val):,}  test={len(test):,}")

    # ── Save graph cache to Drive ─────────────────────────────────────────────
    if graphs_cache_dir is not None:
        cache_file = Path(graphs_cache_dir) / f"{tag}.pkl"
        print(f"[data_loader] Saving cache → {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump((train, val, test), f, protocol=4)
        print("[data_loader] Cache saved ✔")

    return train, val, test
