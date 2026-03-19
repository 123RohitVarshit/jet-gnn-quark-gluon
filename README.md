# 🔬 Quark-Gluon Jet Tagging with Graph Neural Networks

> *Can a neural network learn the same physics that took decades for particle physicists to understand?*

This project trains two Graph Neural Networks to classify high-energy particle jets as coming from **Quarks** or **Gluons** — a fundamental problem in experimental particle physics at the Large Hadron Collider (LHC).

---

## 📖 What is this project about?

### The Physics Problem (explained simply)

When two protons collide at the LHC at enormous energies, the quarks and gluons inside them get knocked out. But a bare quark or gluon cannot exist alone in nature — it immediately sprays into a **cone of hundreds of particles** all flying in roughly the same direction. This spray is called a **jet**.

```
Proton collision
      ↓
  Quark/Gluon knocked out
      ↓
  Sprays into hundreds of particles ──→  This is a JET
      ↓
  Detector records each particle as (pT, η, φ, type)
```

The challenge: **given only the list of particles in a jet, can you tell whether it came from a quark or a gluon?**

This matters because:
- Quarks and gluons behave differently under the strong nuclear force
- Being able to tag them helps physicists search for new particles and test the Standard Model
- It is one of the hardest classification problems in particle physics

### Why it is hard

Quark and gluon jets look similar but have subtle statistical differences:

| Property | Quark Jet | Gluon Jet |
|---|---|---|
| Shape | Narrow, concentrated | Broad, diffuse |
| Particles | Fewer | More |
| Energy | Focused in hard core | Spread across jet |
| Radiation | Less soft radiation | More soft radiation (pileup) |

No single particle tells you the answer. The classifier must learn the **collective pattern** across the whole jet.

---

## 🧠 Why Graph Neural Networks?

A jet is a **set of particles** — unordered, variable in size, and existing in continuous space. Standard neural network approaches fail here:

| Method | Problem |
|---|---|
| RNN | Forces artificial sequence — particle order is physically meaningless |
| CNN | Requires fixed pixel grid — destroys kinematic precision, wastes memory |
| Plain MLP | Cannot handle variable-length input |
| **GNN** | ✅ Handles unordered sets natively, variable size, continuous space |

A Graph Neural Network treats each particle as a **node** and connects nearby particles with **edges**, then passes messages between them to learn collective patterns.

---

## 📐 How a Jet Becomes a Graph

Each jet is converted from a raw list of particles into a graph through these steps:

### Step 1 — Node Features
Each particle becomes a node with 6 features:

| Feature | Formula | Why |
|---|---|---|
| `pt_rel` | pT / jet pT | Fractional momentum — scale invariant |
| `eta_rel` | η − η_jet | Position relative to jet centre |
| `phi_rel` | φ − φ_jet | Angular position relative to centre |
| `log_pt` | log(pT) | Compresses large momentum range |
| `log_E` | log(pT × cosh(η)) | Approximate energy |
| `pdgid` | encoded particle type | Photon, pion, kaon, proton, etc. |

### Step 2 — Edge Construction (k-NN)
Connect each particle to its **16 nearest neighbours** in (η, φ) space. This captures physical proximity in the detector.

### Step 3 — Edge Features
Each connection gets 4 features describing the relationship between particles:

| Feature | Meaning |
|---|---|
| `Δη` | Rapidity separation |
| `Δφ` | Angular separation |
| `ΔR` | Total angular distance √(Δη² + Δφ²) |
| `log_kt` | kt distance — momentum-weighted separation |

```
Raw particle list          →        Graph
──────────────────────────────────────────
[pt, η, φ, pdgid]          →   Nodes + Edges
    × 50-150 particles          + 6 node features
                                + 4 edge features
                                + k-NN connections
```

---

## 🏗️ The Two Architectures

### Architecture 1 — ParticleNet (Dynamic EdgeConv)

**Philosophy**: *Change who you talk to based on what you have learned*

```
Layer 1: Build graph in (η, φ) space
         Each particle talks to physical neighbours
         → Learns local subjet structure
              ↓
Layer 2: THROW AWAY the graph
         Rebuild graph in learned feature space
         → Particles connect based on similarity, not position
         → Discovers long-range quantum correlations
              ↓
Layer 3: Rebuild graph again
         → Even deeper abstract correlations
              ↓
Global pooling → Classifier
```

**Key operation — EdgeConv:**
```
message(i→j) = MLP( [features_i,  features_j − features_i] )
                          ↑                    ↑
                   absolute info        relative info
```

The `features_j − features_i` term is crucial — it captures not just what each particle is, but *how different* neighbouring particles are from each other.

**Why this matters for physics**: A single hard-scattered quark might produce two energetic particles flying far apart in the detector. They are physically distant but quantum-mechanically correlated. Dynamic rewiring connects them in feature space even when physical proximity would not.

---

### Architecture 2 — JetGAT (Graph Attention Network v2)

**Philosophy**: *Keep the same connections but learn how loud each voice is*

```
Build graph ONCE in (η, φ) space → Never changes

Layer 1: Every particle listens to neighbours
         BUT each connection has a learned attention weight α_ij
         
         100 GeV hard core particle → α = 0.8 (loud)
         0.5 GeV pileup particle    → α = 0.02 (muted)
              ↓
Layer 2: Same graph, updated attention weights
              ↓
Layer 3: Same graph, updated attention weights
              ↓
Global pooling → Classifier
```

**Key operation — Attention:**
```
α_ij = softmax( attention_score(features_i, features_j, edge_features) )

new_features_i = Σ α_ij × features_j   for all neighbours j
                  ↑
         weighted sum — important particles contribute more
```

**Why this matters for physics**: Jets are polluted with soft radiation and pileup from overlapping collisions. Not all particles carry useful classification information. GAT automatically learns to ignore the noise.

---

### The Fundamental Difference

| | ParticleNet | JetGAT |
|---|---|---|
| Graph | Rebuilt every layer | Fixed from start |
| Aggregation | Equal weight MAX | Learned attention |
| Key strength | Long-range correlations | Noise filtering |
| Compute cost | High (3× graph builds) | Low (1× graph build) |
| Parameters | ~1,450,000 | ~369,000 |
| Question it answers | *Who should talk to whom?* | *How loud should each voice be?* |

---

## 📊 Results

### Dataset
- **Source**: Zenodo record 3164691 — Pythia8 Q/G jets
- **Size**: 100,000 jets (50,000 quark + 50,000 gluon, stratified from all 20 files)
- **Split**: 70% train / 15% val / 15% test
- **Hardware**: Google Colab T4 GPU

### Performance on Pythia8 (Training Distribution)

| Model | Test AUC | Test Accuracy | Parameters | Time/epoch |
|---|---|---|---|---|
| ParticleNet | 0.8877 | 81.13% | 1,450,000 | ~29s |
| **JetGAT** | **0.8953** | **81.97%** | **369,000** | faster |

**JetGAT wins with 4× fewer parameters.**

### Cross-Generator Robustness (Herwig — Never Seen During Training)

| Model | Herwig AUC | AUC Drop |
|---|---|---|
| ParticleNet | 0.7992 | −0.0885 |
| **JetGAT** | **0.8036** | −0.0917 |

Both models were trained only on Pythia8 jets and tested zero-shot on Herwig jets — a fundamentally different Monte Carlo generator with different hadronization physics. JetGAT maintains its advantage.

### Background Rejection (Physics Convention)

In particle physics the key metric is: *"If I keep X% of quarks, how many gluons do I reject?"*

| Signal Efficiency ε_S | ParticleNet 1/ε_B | JetGAT 1/ε_B |
|---|---|---|
| 30% | higher | higher |
| 50% | ~30 | ~33 |
| 70% | moderate | moderate |

Higher 1/ε_B = better background rejection.

### Key Findings

**1. JetGAT wins on every primary metric**
Better AUC, better accuracy, better cross-generator generalisation, 4× fewer parameters.

**2. Static attention beats dynamic rewiring for this task**
For quark-gluon classification, the dominant challenge is suppressing soft pileup radiation — exactly what attention is designed for. Dynamic graph rewiring (ParticleNet's strength) is less useful here because the important correlations are local and captured by the initial physical graph.

**3. Both models generalise reasonably across generators**
~0.09 AUC drop is consistent with published literature. The degradation is mainly caused by the large difference in jet multiplicity between generators (Herwig max 136 particles vs Pythia max 556 particles).

**4. AUC vs Accuracy split on Herwig**
ParticleNet has higher accuracy at fixed threshold on Herwig (0.7109 vs 0.6864) but lower AUC (0.7992 vs 0.8036). This indicates threshold miscalibration under distribution shift — AUC is the correct metric for cross-generator comparison.

---

## 📁 Project Structure

```
jet-gnn-quark-gluon/
│
├── README.md                    ← You are here
├── requirements.txt             ← All dependencies
│
├── models/
│   ├── __init__.py              ← Package init
│   ├── particle_net.py          ← ParticleNet — DynamicEdgeConv
│   └── jet_gat.py               ← JetGAT — GATv2 attention
│
├── data_loader.py               ← Dataset download + graph construction
├── trainer.py                   ← Training loop with checkpointing
│
├── results/
│   ├── roc_curves.png           ← ROC comparison
│   ├── training_curves.png      ← Loss/AUC/Acc over epochs
│   ├── confusion_matrices.png   ← Where models make mistakes
│   ├── score_distributions.png  ← Quark/gluon score separation
│   ├── cross_generator.png      ← Pythia vs Herwig robustness
│   └── graph_construction.png   ← Point-cloud to graph visualisation
│
└── notebooks/
    └── JetGNN_Colab.ipynb       ← Complete runnable notebook
```

---

## 🚀 How to Run

### Option A — Google Colab (Recommended)
Open `notebooks/JetGNN_Colab.ipynb` in Google Colab with a T4 GPU runtime.
The notebook handles everything automatically including dataset download,
graph construction, training, and evaluation.

### Option B — Local

**1. Install dependencies**
```bash
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install energyflow scikit-learn matplotlib numpy==1.26.4
```

**2. Train ParticleNet**
```python
from data_loader import load_dataset
from trainer import train_model

train_data, val_data, test_data = load_dataset(n_per_class=50_000)

history, test_auc = train_model(
    model_name = 'particlenet',
    train_data = train_data,
    val_data   = val_data,
    test_data  = test_data,
    ckpt_dir   = 'checkpoints',
    log_dir    = 'logs',
    epochs     = 30,
    batch_size = 128,
)
print(f'ParticleNet AUC: {test_auc:.4f}')
```

**3. Train JetGAT**
```python
history, test_auc = train_model(
    model_name = 'jetgat',
    train_data = train_data,
    val_data   = val_data,
    test_data  = test_data,
    ckpt_dir   = 'checkpoints',
    log_dir    = 'logs',
    epochs     = 30,
    batch_size = 256,
)
print(f'JetGAT AUC: {test_auc:.4f}')
```

---

## 📦 Dataset

| Property | Value |
|---|---|
| Source | Zenodo record 3164691 |
| Generator | Pythia8, √s = 14 TeV LHC |
| Total size | 2M jets (20 files × 100k) |
| Used | 100k jets (stratified sample) |
| Labels | quark=1, gluon=0 |
| Cross-gen test | Herwig, Zenodo record 3066475 |

The dataset is downloaded automatically via the `energyflow` package on first run and cached locally.

**Why stratified sampling?**
Each of the 20 files is an independent Pythia8 run with slightly different pT and multiplicity distributions. Taking a small slice from every file captures the full kinematic diversity of the generator, whereas loading only the first few files biases training toward those runs' specific kinematics.

---

## 🔧 Technical Details

| Component | Choice | Reason |
|---|---|---|
| Optimizer | AdamW | Weight decay regularisation |
| Loss | CrossEntropyLoss (label smoothing 0.05) | Prevents overconfident predictions |
| LR schedule | Cosine annealing + 3 epoch warmup | Stable training |
| Mixed precision | FP16 via torch.amp | Faster training on T4 |
| Gradient clipping | max norm = 1.0 | Training stability |
| Early stopping | patience = 7 epochs on val AUC | Prevents overfitting |
| Graph construction | k-NN, k=16 in (η, φ) | Physical proximity |
| Max particles | 150 | Covers 99% of jets |

---

## 📚 References

1. **ParticleNet**: Qu & Gouskos — *ParticleNet: Jet Tagging via Particle Clouds*  
   Phys. Rev. D 101, 056019 (2020). [arXiv:1902.08570](https://arxiv.org/abs/1902.08570)

2. **GATv2**: Brody, Alon & Yahav — *How Attentive are Graph Attention Networks?*  
   ICLR 2022. [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)

3. **Dataset**: Komiske, Metodiev & Thaler  
   [Zenodo record 3164691](https://zenodo.org/records/3164691) (2019)

4. **EnergyFlow**: Komiske, Metodiev & Thaler — *Energy Flow Networks: Deep Sets for Particle Jets*  
   JHEP 01 (2019) 121. [arXiv:1810.05163](https://arxiv.org/abs/1810.05163)

---

## 👤 Author

**Rohit** — B.E. Artificial Intelligence & Machine Learning  
SIES Graduate School of Technology, Navi Mumbai  

Research interests: Graph Neural Networks, Evolutionary Algorithms, LLMs, Quantum Computing

---

*Built as part of a particle physics ML project comparing GNN architectures for jet tagging.*
