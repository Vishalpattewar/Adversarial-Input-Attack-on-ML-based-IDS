# Adversarial Input Attack on ML-based Intrusion Detection System
### CICIDS2017 Dataset | CDAC ITISS Bengaluru

> Demonstrating vulnerability of ML-based IDS to adversarial
> attacks (FGSM + PGD) and adversarial training as an effective
> defence — using the modern CICIDS2017 network traffic dataset.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Results](#-results)
4. [Project Structure](#-project-structure)
5. [Requirements](#-requirements)
6. [Installation](#-installation)
7. [Dataset Setup](#-dataset-setup)
8. [Run Experiment](#-run-experiment)
9. [View Results](#-view-results)
10. [Live IDS Demo](#-live-ids-demo)
11. [Interface Guide](#-interface-guide)
12. [Troubleshooting](#-troubleshooting)
13. [Academic References](#-academic-references)

---

## 🎯 Project Overview

This project demonstrates that Machine Learning-based Intrusion
Detection Systems (IDS) are **vulnerable to adversarial attacks**
and that **adversarial training** is an effective defence.

### What This Project Does

- Trains a neural network (IDSNet) to detect network attacks on the CICIDS2017 dataset (52 features, 100,000 samples)
- Attacks the trained model using:
  - → FGSM (Fast Gradient Sign Method) — single-step attack
  - → PGD (Projected Gradient Descent) — multi-step attack
  - Only perturbs features the attacker can realistically control
- Trains an adversarially-hardened model using FGSM augmentation
- Shows it resists attacks significantly better
- Compares standard vs adversarial model under attack
- Generates evasion curves and defence gain charts
- Demonstrates real-time IDS with two modes:
  - → `live_ids.py` — per-packet (NSL-KDD model)
  - → `live_ids_cicids.py` — per-flow with flow accumulator (CICIDS2017 model, replicates CICFlowMeter)

### Why CICIDS2017

- **NSL-KDD (1999)** → Classic benchmark, simulated traffic
- **CICIDS2017 (2017)** → Modern real network captures
- DoS, DDoS, BruteForce, WebAttacks, Botnet, PortScan — current attack types
- Industry standard for IDS research

### Academic Contribution

- Proves ML-based IDS vulnerability on a **modern dataset**
- Implements **realistic adversarial attack** (mutable features only)
- Demonstrates **adversarial training** reduces evasion significantly
- Implements **flow-level live IDS** replicating CICFlowMeter logic

---

## 🏗️ Architecture

### IDSNet Neural Network

```
Input (52 features)
      ↓
 BatchNorm1d (normalise feature scales)
      ↓
 Dense (52 → 256) + ReLU + Dropout(0.3)
      ↓
 Dense (256 → 128) + ReLU + Dropout(0.3)
      ↓
 Dense (128 → 64)  + ReLU
      ↓
 Dense (64 → 1)    + Sigmoid
      ↓
Output: Attack probability [0.0 → 1.0]
        >= 0.5 = Attack, < 0.5 = Normal
```

**Total parameters:** ~54,785  
**Loss:** Binary Cross Entropy  
**Optimiser:** Adam (lr=1e-3, weight_decay=1e-4)  
**Scheduler:** StepLR (step=10, γ=0.5)

### Attack Configuration

```
FGSM: x_adv = x + ε · sign(∇_x Loss(f(x), y))
PGD : x_t+1 = Proj(x_t + α · sign(∇_x Loss(f(x_t), y)))
      α = ε/10 (10 steps)

Epsilon range  : [0.05, 0.10, 0.20, 0.50, 0.75, 1.00, 1.50, 2.00]
Mutable features: 27 out of 52 (attacker-controllable only)
Physical constraint: all features clipped to >= 0
```

---

## 📊 Results

> Results populated after running `run_experiment.py`

| Model        | Accuracy  | Max FGSM Evasion | Max PGD Evasion |
|--------------|-----------|------------------|-----------------|
| Standard     | ~96-98%   | ~80-90%          | ~85-95%         |
| Adversarial  | ~94-96%   | ~40-60%          | ~45-65%         |

**Defence gain:** Up to +40% reduction in evasion rate

**Key finding:** Adversarial training (ε=0.80) significantly
hardens the IDS against both FGSM and PGD attacks with only
~1-2% accuracy trade-off on clean traffic.

---

## 📁 Project Structure

```
adversarial_ids_cicids/
│
├── prepare_cicids.py          # One-time data preprocessing
│                              # Loads cleaned CSV → 100K sample CSV
│
├── src/
│   ├── ids_model.py           # IDSNet architecture + IDSTrainer
│   │                          # Defines 52 CICIDS2017 features
│   │                          # Training, saving, loading, predicting
│   │
│   ├── fgsm_pgd_attack.py     # FGSM + PGD adversarial attacks
│   │                          # Defines 27 mutable features
│   │                          # Epsilon sweep evaluation
│   │
│   ├── run_experiment.py      # Main pipeline orchestrator
│   │                          # Train → Attack → Compare → Save
│   │
│   ├── visualise_results.py   # Generates 3 result plots
│   │                          # Evasion curves, defence gain,
│   │                          # architecture diagram
│   │
│   ├── live_ids.py            # Real-time per-packet IDS
│   │                          # Uses NSL-KDD model (38 features)
│   │                          # Instant alerts via Scapy
│   │
│   └── live_ids_cicids.py     # Real-time per-flow IDS
│                              # Uses CICIDS2017 model (52 features)
│                              # Flow accumulator (CICFlowMeter-like)
│                              # Alerts after flow completes
│
├── test_traffic.py            # Synthetic traffic generator
│                              # Sends normal + attack packets
│                              # to localhost for live IDS demo
│
├── data/
│   ├── README_DATASET.md      # Dataset download instructions
│   └── cicids2017/            # Place downloaded CSV here
│       └── cicids2017_cleaned.csv  ← download from Kaggle
│
├── models/
│   └── cicids/                # Generated after training
│       ├── ids_standard.pth
│       ├── ids_adversarial.pth
│       ├── scaler_standard.pkl
│       ├── scaler_adversarial.pkl
│       ├── metrics_standard.json
│       ├── metrics_adversarial.json
│       ├── history_standard.json
│       └── history_adversarial.json
│
├── results/
│   └── cicids/                # Generated after experiment
│       ├── standard/
│       │   ├── attack_results_fgsm.json
│       │   └── attack_results_pgd.json
│       ├── adversarial/
│       │   ├── attack_results_fgsm.json
│       │   └── attack_results_pgd.json
│       ├── plots/
│       │   ├── evasion_curves.png
│       │   ├── defence_gain.png
│       │   └── architecture.png
│       └── summary_report.json
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## 💻 Requirements

### Operating System

| OS                          | Supported        | Notes                          |
|-----------------------------|------------------|--------------------------------|
| Ubuntu 20.04+               | ✅ Full support   | Recommended                    |
| WSL2 (Ubuntu on Windows 11) | ✅ Full support   | Use `lo` interface             |
| Kali Linux                  | ✅ Full support   | sudo required for Scapy        |
| Debian / Linux Mint         | ✅ Full support   |                                |
| macOS 12+                   | ✅ Full support   | Use `lo0` interface            |
| Windows 11 (native)         | ⚠️ Partial        | No Scapy live capture          |
| Windows + WSL2              | ✅ Full support   | Recommended for Windows        |

### Python

- Python 3.9 or higher
- Python 3.12.x recommended (tested on 3.12.3)

### Key Dependencies

| Package      | Version   | Purpose                          |
|--------------|-----------|----------------------------------|
| torch        | 2.12.1+cpu| Neural network                   |
| numpy        | 1.26.4    | Numerical operations             |
| pandas       | 2.2.2     | Data loading                     |
| scikit-learn | 1.4.2     | StandardScaler, metrics          |
| matplotlib   | 3.9.0     | Plot generation                  |
| scapy        | 2.5.0     | Packet capture (live IDS)        |
| rich         | 13.7.1    | Terminal dashboard               |
| joblib       | 1.4.2     | Model persistence                |

---

## ⚙️ Installation

Choose your operating system:

---

### 🐧 Ubuntu / Debian / Linux Mint

```bash
# Step 1 — Update system
sudo apt update && sudo apt upgrade -y

# Step 2 — Install Python and system dependencies
sudo apt install -y python3 python3-pip python3-venv git

# Step 3 — Clone repository
git clone https://github.com/Vishalpattewar/adversarial-ids-cicids.git
cd adversarial-ids-cicids

# Step 4 — Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Step 5 — Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Step 6 — Install PyTorch (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 7 — Verify installation
python3 -c "
import torch, numpy, pandas, sklearn, matplotlib, scapy, rich, joblib
print('All packages OK')
print(f'PyTorch version: {torch.__version__}')
"
```

### 🪟 WSL2 (Ubuntu on Windows 11)

```bash
# Step 1 — Open WSL terminal (Ubuntu)
# Press Win+R → type 'wsl' → Enter

# Step 2 — Update WSL Ubuntu
sudo apt update && sudo apt upgrade -y

# Step 3 — Install dependencies
sudo apt install -y python3 python3-pip python3-venv git

# Step 4 — Clone repository
cd ~
git clone https://github.com/Vishalpattewar/adversarial-ids-cicids.git
cd adversarial-ids-cicids

# Step 5 — Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Step 6 — Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 7 — Verify
python3 -c "
import torch, numpy, pandas, sklearn, matplotlib, scapy, rich, joblib
print('All packages OK')
"

# NOTE FOR WSL — Live IDS interface:
# Use INTERFACE = 'lo' in live_ids.py and live_ids_cicids.py
# WSL does not reliably auto-detect interfaces (INTERFACE=None)
# 'lo' = loopback = captures test_traffic.py demo packets
```

### 🐉 Kali Linux

```bash
# Step 1 — Update Kali
sudo apt update && sudo apt full-upgrade -y

# Step 2 — Kali has Python pre-installed, install venv
sudo apt install -y python3-venv git

# Step 3 — Clone repository
git clone https://github.com/Vishalpattewar/adversarial-ids-cicids.git
cd adversarial-ids-cicids

# Step 4 — Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Step 5 — Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 6 — Scapy on Kali may need extra library
sudo apt install -y libpcap-dev

# Step 7 — Verify
python3 -c "
import torch, scapy, rich
print('Kali setup OK')
"

# NOTE FOR KALI — Live IDS:
# Kali has multiple interfaces — check yours:
# ip link show
# Common: eth0 (wired), wlan0 (wireless), lo (loopback)
# For demo: INTERFACE = 'lo'
# For real traffic capture: INTERFACE = 'eth0' or 'wlan0'
```

### 🍎 macOS

```bash
# Step 1 — Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Step 2 — Install Python
brew install python@3.12 git

# Step 3 — Clone repository
git clone https://github.com/Vishalpattewar/adversarial-ids-cicids.git
cd adversarial-ids-cicids

# Step 4 — Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Step 5 — Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install torch

# Step 6 — Verify
python3 -c "
import torch, scapy, rich
print('macOS setup OK')
"

# NOTE FOR macOS — Live IDS:
# Check interfaces: ifconfig
# Use INTERFACE = 'lo0' for loopback demo
# Use INTERFACE = 'en0' for WiFi real traffic
# sudo is required for Scapy on macOS
```

### 🪟 Windows 11 (Native — Limited)

```powershell
# NOTE: Live IDS (Scapy) does not work reliably on Windows native
# Recommended: use WSL2 for full functionality
# For experiment + visualisation only (no live IDS):

# Step 1 — Install Python 3.12 from python.org
# Download: https://www.python.org/downloads/

# Step 2 — Open Command Prompt or PowerShell

# Step 3 — Clone repository
git clone https://github.com/Vishalpattewar/adversarial-ids-cicids.git
cd adversarial-ids-cicids

# Step 4 — Create virtual environment
python -m venv venv
venv\Scripts\activate

# Step 5 — Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 6 — Install Npcap for Scapy (if using live IDS)
# Download: https://npcap.com/#download
# Install with "WinPcap API-compatible mode" checked

# Step 7 — Verify
python -c "import torch, numpy, pandas; print('Windows setup OK')"
```

---

## 📦 Dataset Setup

The CICIDS2017 cleaned dataset must be downloaded manually from Kaggle.

### Step 1 — Download Dataset

```
Dataset  : CICIDS2017 Cleaned and Preprocessed
Source   : Kaggle
URL      : https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed
File     : cicids2017_cleaned.csv
Size     : 685 MB
Rows     : 2,520,751
Features : 52 numeric + 1 label column
```

#### Option A — Kaggle CLI (Recommended)

```bash
# Install Kaggle CLI
pip install kaggle

# Setup Kaggle API key
# 1. Go to: kaggle.com → Account → Settings → API → Create New Token
# 2. Downloads kaggle.json to your Downloads folder

# Linux / WSL / macOS:
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows:
# Copy kaggle.json to C:\Users\YourName\.kaggle\

# Download dataset
mkdir -p data/cicids2017
cd data/cicids2017
kaggle datasets download -d ericanacletoribeiro/cicids2017-cleaned-and-preprocessed
unzip *.zip
cd ../..
```

#### Option B — Manual Browser Download

```
1. Open browser → go to:
   https://www.kaggle.com/datasets/ericanacletoribeiro/
   cicids2017-cleaned-and-preprocessed

2. Click Download button
   (requires free Kaggle account)

3. Save downloaded file

4. Move to project:
   Linux / WSL / macOS:
     mkdir -p data/cicids2017
     mv ~/Downloads/cicids2017_cleaned.csv data/cicids2017/

   Windows:
     Create folder: adversarial-ids-cicids\data\cicids2017\
     Copy cicids2017_cleaned.csv into that folder
```

### Step 2 — Verify Download

```bash
# Check file exists and size
ls -lh data/cicids2017/cicids2017_cleaned.csv
# Expected: ~685MB

# Check row count
wc -l data/cicids2017/cicids2017_cleaned.csv
# Expected: 2520752 (2,520,751 rows + 1 header)

# Check columns
head -1 data/cicids2017/cicids2017_cleaned.csv
# Expected: Destination Port,Flow Duration,...,Attack Type
```

### Step 3 — Prepare Dataset

```bash
# Navigate to project root
cd ~/adversarial-ids-cicids   # adjust path for your system

# Activate venv
source venv/bin/activate      # Linux / WSL / macOS
# OR
venv\Scripts\activate         # Windows

# Run preparation script (takes 2-5 minutes)
python3 prepare_cicids.py

# Expected output:
# [1/7] Loading CSV file...                       Loaded in Xs
# [2/7] Validating NaN values...                  No NaN values found ✓
# [3/7] Validating infinite values...             No infinite values found ✓
# [4/7] Original label distribution:
#   Normal Traffic    2,095,057  (83.1%)
#   DoS                 193,745  ( 7.7%)
#   DDoS                128,014  ( 5.1%)
#   Port Scanning        90,694  ( 3.6%)
#   Brute Force           9,150  ( 0.4%)
#   Web Attacks           2,143  ( 0.1%)
#   Bots                  1,948  ( 0.1%)
# [5/7] Converting to binary labels...
# [6/7] Stratified sampling to 100,000 rows...
# [7/7] Validating output...
# Preprocessing Complete
# Output: data/network_flows_cicids.csv (100,000 rows)

# Verify output
wc -l data/network_flows_cicids.csv
# Expected: 100001 (100,000 rows + 1 header)
```

---

## 🚀 Run Experiment

```bash
# Navigate to src/
cd src

# Activate venv (if not active)
source ../venv/bin/activate   # Linux / WSL / macOS
# OR
..\venv\Scripts\activate      # Windows

# Run full experiment pipeline
python3 run_experiment.py

# ── What it does: ────────────────────────────────────────────
# STEP 1: Load network_flows_cicids.csv (100K rows, 52 features)
# STEP 2: Train STANDARD model     (~15-20 minutes)
# STEP 3: Train ADVERSARIAL model  (~20-25 minutes)
# STEP 4: Attack standard model    (FGSM + PGD sweep, ~5 minutes)
# STEP 5: Attack adversarial model (FGSM + PGD sweep, ~5 minutes)
# STEP 6: Compare + save summary   (~1 minute)
# ─────────────────────────────────────────────────────────────

# Expected time: 30-50 minutes on CPU

# Expected output snippet:
# Training Standard IDS Model
#   Dataset      : CICIDS2017
#   Architecture : 52→256→128→64→1
#   Epoch [ 5/30] Loss: 0.xxxx | Val Acc: 96.xx%
#   ...
#   Epoch [30/30] Loss: 0.xxxx | Val Acc: 97.xx%
#
# Training Adversarially-Hardened IDS Model
#   FGSM epsilon : 0.8
#   Epoch [ 5/30] Loss: 0.xxxx | Val Acc: 95.xx%
#   ...
#
# FGSM Epsilon Sweep
#   ε=0.05   Clean: 97.x%  | Adv: 97.x%  | Evasion:  2.x%
#   ε=0.5    Clean: 97.x%  | Adv: 75.x%  | Evasion: 25.x%
#   ε=1.0    Clean: 97.x%  | Adv: 25.x%  | Evasion: 75.x%
#   ε=2.0    Clean: 97.x%  | Adv:  5.x%  | Evasion: 95.x%
#
# Experiment Complete
# Models saved : models/cicids/
# Results saved: results/cicids/

# If training too slow (>60 minutes):
# Edit src/run_experiment.py line: EPOCHS = 20  (reduce from 30)
# Re-run → 20 epochs sufficient for research results

# If memory error:
# Edit prepare_cicids.py line: N_SAMPLES = 50_000 (reduce from 100K)
# Re-run prepare_cicids.py then run_experiment.py
```

---

## 📈 View Results

```bash
# Generate 3 plots from results
cd src   # if not already there
python3 visualise_results.py

# Generated files:
# results/cicids/plots/evasion_curves.png  ← evasion vs epsilon
# results/cicids/plots/defence_gain.png    ← defence improvement
# results/cicids/plots/architecture.png    ← IDSNet diagram
```

### Open Plots

```bash
# WSL2 (Ubuntu on Windows):
explorer.exe ../results/cicids/plots/

# Ubuntu Desktop:
xdg-open ../results/cicids/plots/

# Kali Linux:
eog ../results/cicids/plots/*.png
# OR
display ../results/cicids/plots/evasion_curves.png

# macOS:
open ../results/cicids/plots/

# Windows (PowerShell):
start ..\results\cicids\plots\
```

### View Summary Report

```bash
cat ../results/cicids/summary_report.json
# Shows accuracy, evasion rates, defence gains
```

---

## 🔴 Live IDS Demo

Two live IDS modes are available. Run `run_experiment.py` first to train models.

### Mode 1 — Per-Packet IDS (NSL-KDD Model)

Fast per-packet detection. Instant alerts. Uses NSL-KDD adversarially-hardened model (38 features).

```bash
# Terminal 1 — Start Live IDS
cd ~/adversarial-ids-cicids/src   # adjust path

# Linux / WSL:
sudo /full/path/to/venv/bin/python3 live_ids.py

# Example WSL full path:
sudo /home/YOUR_USERNAME/adversarial-ids-cicids/venv/bin/python3 live_ids.py

# macOS:
sudo ../venv/bin/python3 live_ids.py

# Windows (PowerShell as Administrator):
python live_ids.py
```

```bash
# Terminal 2 — Send Test Traffic (open new terminal)
cd ~/adversarial-ids-cicids

# Linux / WSL / macOS:
sudo /full/path/to/venv/bin/python3 test_traffic.py

# Choose option:
# 1 → Normal traffic only
# 2 → Attack traffic only   ← triggers alerts
# 3 → Mixed traffic         ← realistic demo
# 4 → Continuous loop
```

```
Dashboard shows:
  ┌─────────────────────────────────────────────┐
  │   Adversarial IDS — Live Network Monitor    │
  ├─────────────────┬───────────────────────────┤
  │  Live Statistics│  Recent Alerts            │
  │  Packets: 902   │  192.168.x.x → 127.0.0.1  │
  │  Normal : 742   │  TCP | 99.1% | HIGH        │
  │  Attacks: 160   │  ...                      │
  └─────────────────┴───────────────────────────┘
```

### Mode 2 — Per-Flow IDS (CICIDS2017 Model)

Flow-level detection using flow accumulator. Replicates CICFlowMeter feature extraction.
Alerts appear after flow completes (2-30 second delay).

```bash
# Terminal 1 — Start CICIDS Flow IDS
cd ~/adversarial-ids-cicids/src

sudo /full/path/to/venv/bin/python3 live_ids_cicids.py
```

```bash
# Terminal 2 — Send Test Traffic (same as Mode 1)
sudo /full/path/to/venv/bin/python3 ../test_traffic.py
# Choose option 3 (mixed)
```

> **Note:** Alerts appear after flow ends.  
> Short flows timeout after `FLOW_TIMEOUT=30` seconds.  
> This is expected behaviour for flow-level IDS.

---

## 🌐 Interface Guide

The network interface must match your OS and use case.

### Check Your Interface

```bash
# Linux / WSL / Kali:
ip link show

# macOS:
ifconfig

# Common interface names:
# lo / lo0   = loopback  → for local demo only
# eth0       = wired Ethernet
# wlan0      = WiFi (Linux)
# en0        = WiFi (macOS)
# ens33      = VMware virtual NIC
# enp0s3     = VirtualBox virtual NIC
```

### Interface Selection Per OS

| OS           | Demo with `test_traffic.py` | Real Network Capture      |
|--------------|-----------------------------|---------------------------|
| WSL2         | `'lo'`                      | `'eth0'`                  |
| Ubuntu       | `'lo'`                      | `'eth0'` or `'enp0s3'`    |
| Kali Linux   | `'lo'`                      | `'eth0'` or `'wlan0'`     |
| macOS        | `'lo0'`                     | `'en0'`                   |
| Windows      | `'lo'`                      | N/A (use WSL2)            |

### How To Change Interface

```bash
# Edit live_ids.py OR live_ids_cicids.py
# Find this line near top of file:

INTERFACE = 'lo'   # for demo with test_traffic.py
# OR
INTERFACE = None   # auto-detect (not reliable on WSL)
# OR
INTERFACE = 'eth0' # real network capture

# Change to match your interface name
# from 'ip link show' output
```

---

## 🔧 Troubleshooting

### Installation Issues

```
ERROR: ModuleNotFoundError: No module named 'torch'
FIX:   pip install torch --index-url \
         https://download.pytorch.org/whl/cpu

ERROR: ModuleNotFoundError: No module named 'scapy'
FIX:   pip install scapy==2.5.0

ERROR: pip install fails with SSL error
FIX:   pip install --trusted-host pypi.org \
         --trusted-host files.pythonhosted.org -r requirements.txt
```

### Dataset Issues

```
ERROR: FileNotFoundError: cicids2017_cleaned.csv not found
FIX:   Download from Kaggle (see Dataset Setup section)
       Save to: data/cicids2017/cicids2017_cleaned.csv

ERROR: CSV has wrong columns
FIX:   Make sure you downloaded the correct dataset:
       "CICIDS2017 Cleaned and Preprocessed"
       URL: https://www.kaggle.com/datasets/
            ericanacletoribeiro/
            cicids2017-cleaned-and-preprocessed

ERROR: network_flows_cicids.csv not found
FIX:   Run prepare_cicids.py first:
       python3 prepare_cicids.py
```

### Experiment Issues

```
ERROR: Training too slow (>60 minutes)
FIX:   Reduce epochs in src/run_experiment.py:
       EPOCHS = 20  (change from 30)

ERROR: MemoryError during training
FIX:   Reduce samples in prepare_cicids.py:
       N_SAMPLES = 50_000  (change from 100_000)
       Re-run prepare_cicids.py then run_experiment.py

ERROR: CUDA/GPU errors
FIX:   This project uses CPU only
       torch is installed as +cpu version
       No GPU configuration needed
```

### Live IDS Issues

```
ERROR: sudo: venv/bin/python3: command not found
FIX:   Use full absolute path:
       sudo /home/USERNAME/adversarial-ids-cicids/venv/bin/python3 \
            live_ids.py

ERROR: OSError: [Errno 19] No such device (interface not found)
FIX:   Check interface name: ip link show
       Change INTERFACE in live_ids.py to correct name
       For WSL use 'lo', for Kali check 'eth0' or 'wlan0'

ERROR: Model not found (live_ids.py)
FIX:   live_ids.py uses NSL-KDD model
       Clone NSL-KDD project alongside this one:
       git clone https://github.com/Vishalpattewar/adversarial-ids
       Run adversarial_ids/src/run_experiment.py

ERROR: Model not found (live_ids_cicids.py)
FIX:   Run CICIDS2017 experiment first:
       cd src && python3 run_experiment.py

ERROR: No alerts in live_ids_cicids.py
FIX:   This is expected — see explanation below
       CICIDS2017 model trained on real bidirectional flows
       test_traffic.py generates simplified packets
       Model correctly classifies them as non-CICIDS-pattern
       Flow timeout is 30 seconds — wait longer
       For best demo: use live_ids.py (NSL-KDD model)

ERROR: 0 packets in live_ids with INTERFACE=None on WSL
FIX:   WSL cannot auto-detect interfaces
       Change INTERFACE = 'lo' in live_ids.py
```

### Plot Issues

```
ERROR: No module named tkinter / display error
FIX:   Already handled — matplotlib uses Agg backend
       Plots saved as PNG files (no display needed)

ERROR: Y-axis shows 0.0% for all values
FIX:   Run run_experiment.py first
       JSON result files must exist before visualising

ERROR: explorer.exe not found (WSL)
FIX:   Use: wslview ../results/cicids/plots/
       OR:  cp results/cicids/plots/*.png /mnt/c/Users/USERNAME/Desktop/
```

---

## 📚 Academic References

```
1. CICIDS2017 Dataset:
   Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. A. (2018).
   Toward Generating a New Intrusion Detection Dataset and
   Intrusion Traffic Characterization.
   ICISSP 2018. DOI: 10.5220/0006639801080116

2. FGSM Attack:
   Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014).
   Explaining and Harnessing Adversarial Examples.
   ICLR 2015. arXiv:1412.6572

3. PGD Attack + Adversarial Training:
   Madry, A., Makelov, A., Schmidt, L., Tsipras, D.,
   & Vladu, A. (2017).
   Towards Deep Learning Models Resistant to Adversarial Attacks.
   ICLR 2018. arXiv:1706.06083

4. CICFlowMeter:
   Lashkari, A. H., Draper-Gil, G., Mamun, M. S. I.,
   & Ghorbani, A. A. (2017).
   Characterization of Tor Traffic Using Time Based Features.
   ICISSP 2017.

5. Adversarial ML on IDS Survey:
   Huang, L., Joseph, A. D., Nelson, B., Rubinstein, B. I.,
   & Tygar, J. D. (2011).
   Adversarial Machine Learning.
   AISec 2011.
```

---

## 👨‍💻 Author

```
Name        : Project-Team
Institution : CDAC ACTS Bengaluru
Project     : Adversarial Input Attack on ML-based IDS
Dataset     : CICIDS2017 (Phase 2)
GitHub      : github.com/Vishalpattewar/Adversarial-Input-Attack-on-ML-based-IDS
```

---

## 📄 License

```
MIT License — Free to use for academic and research purposes.
Please cite this repository and the CICIDS2017 paper if used
in academic work.
```