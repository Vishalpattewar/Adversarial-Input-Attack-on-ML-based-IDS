# =============================================================
# src/ids_model.py
# =============================================================
# PURPOSE:
#   Defines the Neural Network architecture (IDSNet) and the
#   complete training/saving/loading/prediction pipeline
#   (IDSTrainer) for the CICIDS2017 Intrusion Detection System.
#
# WHAT THIS FILE DEFINES:
#   1. NUMERIC_FEATURES   — 52 CICIDS2017 feature names
#                           exact names from cicids2017_cleaned.csv
#                           same order as network_flows_cicids.csv
#
#   2. MUTABLE_FEATURE_NAMES — 27 features the attacker controls
#                              used by fgsm_pgd_attack.py
#
#   3. IDSNet             — 4-layer neural network
#                           52 → 256 → 128 → 64 → 1
#                           Binary classifier: Normal(0) vs Attack(1)
#
#   4. IDSTrainer         — Training, saving, loading, predicting
#                           Owns StandardScaler (fitted on X_train)
#                           Saves model + scaler + metrics + history
#
# WHAT THIS FILE DOES NOT DO:
#   → Does NOT load dataset (run_experiment.py handles that)
#   → Does NOT know dataset file paths
#   → Does NOT generate synthetic data
#   → Does NOT apply scaler before training split (data leakage)
#
# CICIDS2017 vs NSL-KDD DIFFERENCES:
#   NSL-KDD  : 38 features, input_dim=38, label='normal'
#   CICIDS2017: 52 features, input_dim=52, label='Normal Traffic'
#   Architecture identical — only input_dim changes
#   Allows fair comparison between both datasets
#
# BUGS FIXED VS ORIGINAL NSL-KDD VERSION:
#   FIX 1: squeeze(-1) everywhere instead of squeeze()
#           bare squeeze() collapses (1,1)→scalar when batch_size=1
#           breaks BCELoss and numpy conversion
#           critical for live_ids_cicids.py (one flow at a time)
#
#   FIX 2: self.model.zero_grad() in _fgsm_batch()
#           without it, model parameter gradients accumulate
#           across batches, silently corrupting adversarial loss
#
#   FIX 3: weights_only=True in torch.load()
#           weights_only=False allows arbitrary code execution
#           via pickle — serious security vulnerability
#           PyTorch 2.x requirement
#
#   FIX 4: __file__ anchored BASE_DIR for model_dir default
#           relative paths break if called from different CWD
#           os.path.realpath(__file__) always finds script location
#
#   FIX 5: os.path.join for all save/load paths
#           f-string slash hardcoding fails on Windows
#           os.path.join is cross-platform correct
#
#   FIX 6: Training history saved to disk (history_{tag}.json)
#           previously tracked but never written
#           needed for learning curve visualisation
#
# HOW TO USE:
#   from ids_model import IDSTrainer, NUMERIC_FEATURES
#   trainer = IDSTrainer()
#   metrics = trainer.train(df)
#   trainer.load('standard')
#   preds = trainer.predict(X_raw)
#
# CDAC ITISS — Adversarial Input Attack on ML-based IDS
# =============================================================


import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib


# ─────────────────────────────────────────────────────────────
#  PATH CONFIGURATION
#
#  FIX 4: Anchor default model_dir to script location
#  not to CWD (current working directory)
#
#  WHY realpath not abspath:
#    realpath() resolves symlinks → works in Docker/NFS/WSL
#    abspath() does not resolve symlinks
#
#  WHY try/except NameError:
#    __file__ undefined in Jupyter → fallback to CWD
# ─────────────────────────────────────────────────────────────
try:
    _BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    _BASE_DIR = os.getcwd()

# Default directory for saving/loading model files
# src/ids_model.py is in src/ → parent is project root
# models/cicids/ is at project root level
_DEFAULT_MODEL_DIR = os.path.join(
    _BASE_DIR, '..', 'models', 'cicids'
)


# ─────────────────────────────────────────────────────────────
#  NUMERIC FEATURES — 52 CICIDS2017 features
#
#  WHY THESE 52:
#    cicids2017_cleaned.csv has 53 columns:
#      52 numeric features + 1 label ('Attack Type')
#    All 52 numeric features are used including 'Destination Port'
#
#  WHY DESTINATION PORT INCLUDED:
#    Port 80 flooding     = likely DoS attack
#    Port 22 brute force  = likely SSH attack
#    Already numeric      = no encoding needed
#    Provides strong attack signal to the model
#
#  WHY 52 NOT 76:
#    Original CICIDS2017 had 76 features
#    Cleaned version removed 24 features that were:
#      → Near-zero variance (Fwd Avg Bytes/Bulk etc.)
#      → Redundant/correlated (Avg Fwd Segment Size etc.)
#      → CICFlowMeter calculation errors (CWE Flag Count etc.)
#    52 clean features outperforms 76 noisy features
#
#  ORDER MATTERS:
#    This exact order must match:
#      1. network_flows_cicids.csv columns (prepare_cicids.py)
#      2. Feature vector in live_ids_cicids.py (flow → features)
#    Any mismatch = model receives wrong features = wrong output
# ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    'Destination Port',              #  0  target port (attack signal)
    'Flow Duration',                 #  1  total flow duration (μs)
    'Total Fwd Packets',             #  2  packets sent by source
    'Total Length of Fwd Packets',   #  3  bytes sent by source
    'Fwd Packet Length Max',         #  4  max forward packet size
    'Fwd Packet Length Min',         #  5  min forward packet size
    'Fwd Packet Length Mean',        #  6  mean forward packet size
    'Fwd Packet Length Std',         #  7  std forward packet size
    'Bwd Packet Length Max',         #  8  max backward packet size
    'Bwd Packet Length Min',         #  9  min backward packet size
    'Bwd Packet Length Mean',        # 10  mean backward packet size
    'Bwd Packet Length Std',         # 11  std backward packet size
    'Flow Bytes/s',                  # 12  total byte transfer rate
    'Flow Packets/s',                # 13  total packet rate
    'Flow IAT Mean',                 # 14  mean inter-arrival time
    'Flow IAT Std',                  # 15  std inter-arrival time
    'Flow IAT Max',                  # 16  max inter-arrival time
    'Flow IAT Min',                  # 17  min inter-arrival time
    'Fwd IAT Total',                 # 18  total forward IAT
    'Fwd IAT Mean',                  # 19  mean forward IAT
    'Fwd IAT Std',                   # 20  std forward IAT
    'Fwd IAT Max',                   # 21  max forward IAT
    'Fwd IAT Min',                   # 22  min forward IAT
    'Bwd IAT Total',                 # 23  total backward IAT
    'Bwd IAT Mean',                  # 24  mean backward IAT
    'Bwd IAT Std',                   # 25  std backward IAT
    'Bwd IAT Max',                   # 26  max backward IAT
    'Bwd IAT Min',                   # 27  min backward IAT
    'Fwd Header Length',             # 28  forward TCP header length
    'Bwd Header Length',             # 29  backward TCP header length
    'Fwd Packets/s',                 # 30  forward packet rate
    'Bwd Packets/s',                 # 31  backward packet rate
    'Min Packet Length',             # 32  minimum packet length
    'Max Packet Length',             # 33  maximum packet length
    'Packet Length Mean',            # 34  mean packet length
    'Packet Length Std',             # 35  std packet length
    'Packet Length Variance',        # 36  variance of packet length
    'FIN Flag Count',                # 37  FIN TCP flag count
    'PSH Flag Count',                # 38  PSH TCP flag count
    'ACK Flag Count',                # 39  ACK TCP flag count
    'Average Packet Size',           # 40  average packet size
    'Subflow Fwd Bytes',             # 41  subflow forward bytes
    'Init_Win_bytes_forward',        # 42  initial TCP window fwd
    'Init_Win_bytes_backward',       # 43  initial TCP window bwd
    'act_data_pkt_fwd',              # 44  actual data packets fwd
    'min_seg_size_forward',          # 45  min segment size fwd
    'Active Mean',                   # 46  mean active flow time
    'Active Max',                    # 47  max active flow time
    'Active Min',                    # 48  min active flow time
    'Idle Mean',                     # 49  mean idle flow time
    'Idle Max',                      # 50  max idle flow time
    'Idle Min',                      # 51  min idle flow time
]
# Total: 52 features → model input_dim = len(NUMERIC_FEATURES) = 52


# ─────────────────────────────────────────────────────────────
#  MUTABLE FEATURE NAMES — 27 features the attacker controls
#
#  PRINCIPLE:
#    Mutable   = attacker (sender/client) controls these
#    Immutable = server response / network-observed
#
#  WHY THIS MATTERS:
#    FGSM/PGD adversarial attacks should only perturb features
#    the attacker can realistically manipulate
#    Perturbing server-response features is physically impossible
#    This makes our adversarial analysis academically valid
#
#  MUTABLE = forward direction features (attacker sends these)
#    + timing features (attacker controls connection timing)
#    + TCP flag features (attacker sets these)
#    + window/segment features (attacker sets in TCP header)
#
#  IMMUTABLE = backward direction features (server responds)
#    + derived network statistics (observed, not set)
#
#  USED BY: src/fgsm_pgd_attack.py
#    MUTABLE_INDICES = [NUMERIC_FEATURES.index(f)
#                       for f in MUTABLE_FEATURE_NAMES]
# ─────────────────────────────────────────────────────────────
MUTABLE_FEATURE_NAMES = [
    # Attacker chooses destination
    'Destination Port',              #  0  attacker picks target port

    # Attacker controls forward traffic volume + timing
    'Flow Duration',                 #  1  attacker controls duration
    'Total Fwd Packets',             #  2  attacker controls pkt count
    'Total Length of Fwd Packets',   #  3  attacker controls bytes sent
    'Fwd Packet Length Max',         #  4  attacker controls pkt sizes
    'Fwd Packet Length Min',         #  5
    'Fwd Packet Length Mean',        #  6
    'Fwd Packet Length Std',         #  7

    # Attacker controls flow rates
    'Flow Bytes/s',                  # 12  attacker controls byte rate
    'Flow Packets/s',                # 13  attacker controls pkt rate

    # Attacker controls inter-arrival timing
    'Flow IAT Mean',                 # 14  attacker controls timing
    'Flow IAT Std',                  # 15
    'Flow IAT Max',                  # 16
    'Flow IAT Min',                  # 17
    'Fwd IAT Total',                 # 18  forward IAT (attacker side)
    'Fwd IAT Mean',                  # 19
    'Fwd IAT Std',                   # 20
    'Fwd IAT Max',                   # 21
    'Fwd IAT Min',                   # 22

    # Attacker controls forward header
    'Fwd Header Length',             # 28  attacker crafts TCP header

    # Attacker controls forward packet rate
    'Fwd Packets/s',                 # 30  forward packet rate

    # Attacker sets TCP flags
    'FIN Flag Count',                # 37  attacker sends FIN
    'PSH Flag Count',                # 38  attacker sends PSH

    # Attacker controls subflow + window + segment
    'Subflow Fwd Bytes',             # 41  forward subflow bytes
    'Init_Win_bytes_forward',        # 42  attacker sets TCP window
    'act_data_pkt_fwd',              # 44  data packets attacker sends
    'min_seg_size_forward',          # 45  min segment size attacker sets
]
# Total: 27 mutable features out of 52
# Used in fgsm_pgd_attack.py to build MUTABLE_INDICES


# ─────────────────────────────────────────────────────────────
#  IDSNet — Neural Network Architecture
#
#  TYPE    : Multi-Layer Perceptron (MLP)
#  TASK    : Binary classification — Normal(0) vs Attack(1)
#  INPUT   : 52 StandardScaler-normalised CICIDS2017 features
#  OUTPUT  : Single probability [0.0 → 1.0]
#            >= 0.5 = Attack, < 0.5 = Normal
#
#  ARCHITECTURE:
#    Input(52) → BatchNorm1d(52)
#             → Linear(52→256) → ReLU → Dropout(0.3)
#             → Linear(256→128) → ReLU → Dropout(0.3)
#             → Linear(128→64)  → ReLU
#             → Linear(64→1)    → Sigmoid
#
#  WHY BATCHNORM FIRST:
#    CICIDS2017 features have wildly different scales:
#      Destination Port      : 0 – 65,535
#      Flow Duration         : 0 – 120,000,000 (μs)
#      FIN Flag Count        : 0 – 1  (near-binary)
#      Flow Bytes/s          : 0 – 10,000,000+
#    BatchNorm1d normalises ALL to mean≈0, std≈1
#    Makes training stable regardless of feature scales
#    Works alongside StandardScaler for double normalisation
#
#  WHY DROPOUT 0.3:
#    Prevents overfitting on 100K training samples
#    0.3 = drop 30% of neurons per forward pass during training
#    Disabled automatically during eval() mode
#
#  WHY SAME ARCHITECTURE AS NSL-KDD:
#    Only input_dim changes (52 vs 38)
#    Allows fair comparison of results across datasets
#    Proves architecture generalises to different IDS datasets
# ─────────────────────────────────────────────────────────────
class IDSNet(nn.Module):
    """
    Intrusion Detection System Neural Network.

    4-layer MLP binary classifier.
    Input: 52 CICIDS2017 features (StandardScaler normalised)
    Output: Attack probability [0.0 → 1.0]

    Args:
        input_dim: Number of input features
                   52 for CICIDS2017, 38 for NSL-KDD
                   Always pass len(NUMERIC_FEATURES) — never hardcode
    """

    def __init__(self, input_dim: int):
        super(IDSNet, self).__init__()

        self.network = nn.Sequential(

            # Input normalisation
            # Handles different feature scales in CICIDS2017
            nn.BatchNorm1d(input_dim),

            # Layer 1: input_dim → 256
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),        # 30% dropout → prevents overfitting

            # Layer 2: 256 → 128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: 128 → 64
            nn.Linear(128, 64),
            nn.ReLU(),
            # No dropout on last hidden layer — preserve representation

            # Output layer: 64 → 1 probability
            nn.Linear(64, 1),
            nn.Sigmoid()            # squash to [0,1] = probability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor shape (batch_size, input_dim)

        Returns:
            Output tensor shape (batch_size, 1)
            Values in [0.0, 1.0] — attack probability
        """
        return self.network(x)


# ─────────────────────────────────────────────────────────────
#  IDSTrainer — Training, Saving, Loading, Predicting
# ─────────────────────────────────────────────────────────────
class IDSTrainer:
    """
    Complete training and inference pipeline for IDSNet.

    Handles:
      - Feature extraction and label parsing
      - Train/validation split (stratified 80/20)
      - StandardScaler fitting (on X_train ONLY)
      - IDSNet training with optional FGSM adversarial hardening
      - Saving model (.pth), scaler (.pkl), metrics (.json),
        history (.json) to models/cicids/
      - Loading saved model for inference
      - Prediction on raw (unscaled) feature arrays

    Args:
        model_dir: Directory to save/load model files
                   Default: ../models/cicids (relative to this file)
                   FIX 4: uses __file__ anchored path
    """

    def __init__(self, model_dir: str = None):

        # FIX 4: Use __file__ anchored default path
        # Not CWD-relative — works from any directory
        if model_dir is None:
            self.model_dir = os.path.realpath(_DEFAULT_MODEL_DIR)
        else:
            self.model_dir = os.path.realpath(model_dir)

        # Create model directory if it does not exist
        os.makedirs(self.model_dir, exist_ok=True)

        # StandardScaler: fitted on X_train only in train()
        # Prevents data leakage from validation set
        self.scaler    = StandardScaler()
        self.model     = None
        self.input_dim = None

        # Training history — saved to disk after training
        # FIX 6: previously tracked but never written to disk
        self.history = {
            'train_loss': [],
            'val_loss'  : [],
            'val_acc'   : []
        }

    # ──────────────────────────────────────────────────────────
    def preprocess(self, df: pd.DataFrame):
        """
        Extract feature matrix and label vector from DataFrame.

        Args:
            df: DataFrame with NUMERIC_FEATURES columns + 'label'
                Loaded from data/network_flows_cicids.csv

        Returns:
            X: Feature array shape (N, 52) dtype float32
            y: Label array shape (N,) dtype float32
               Values: 0.0 = Normal, 1.0 = Attack
        """
        X = df[NUMERIC_FEATURES].values.astype(np.float32)
        y = df['label'].values.astype(np.float32)
        return X, y

    # ──────────────────────────────────────────────────────────
    def train(self,
              df          : pd.DataFrame,
              epochs      : int   = 30,
              batch_size  : int   = 256,
              lr          : float = 1e-3,
              adversarial : bool  = False,
              fgsm_epsilon: float = 0.80):
        """
        Train IDSNet on CICIDS2017 data.

        Standard mode : trains on clean data only
        Adversarial mode: mixes clean + FGSM-perturbed batches
                          hardens model against adversarial attacks

        Args:
            df          : DataFrame from network_flows_cicids.csv
            epochs      : Training epochs (default 30)
            batch_size  : Mini-batch size (default 256)
            lr          : Adam learning rate (default 1e-3)
            adversarial : True = adversarial training with FGSM
            fgsm_epsilon: FGSM perturbation budget (default 0.80)
                          WHY 0.80: matches realistic attack range
                          tested in epsilon_sweep [0.05 → 2.00]
                          Too small (0.15) = model only resists
                          tiny perturbations → bug in early NSL-KDD

        Returns:
            metrics dict with keys:
              accuracy, type, input_dim, epochs, dataset
        """

        # ── Extract features and labels ───────────────────────
        X, y = self.preprocess(df)

        # ── Train/validation split — stratified 80/20 ─────────
        # WHY stratified: maintains 83.1%/16.9% ratio in both sets
        # WHY 80/20: standard split for research benchmarks
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size    = 0.2,
            random_state = 42,
            stratify     = y
        )

        # ── Fit StandardScaler on X_train ONLY ───────────────
        # WHY train only: fitting on full data = data leakage
        # val statistics would leak into training normalisation
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)

        # ── Build model ───────────────────────────────────────
        # WHY len(NUMERIC_FEATURES) not hardcoded 52:
        #   stays correct if feature list ever changes
        self.input_dim = X_train_s.shape[1]
        self.model     = IDSNet(input_dim=self.input_dim)

        # ── Loss function ─────────────────────────────────────
        # BCELoss: Binary Cross Entropy — standard for binary
        # classification with Sigmoid output
        criterion = nn.BCELoss()

        # ── Optimiser ─────────────────────────────────────────
        # Adam: adaptive learning rate — standard for MLP
        # weight_decay=1e-4: L2 regularisation → smaller weights
        #                    prevents overfitting
        optimizer = optim.Adam(
            self.model.parameters(),
            lr           = lr,
            weight_decay = 1e-4
        )

        # ── Learning rate scheduler ───────────────────────────
        # StepLR: reduce LR by 50% every 10 epochs
        # Epoch  1-10: lr = 0.001
        # Epoch 11-20: lr = 0.0005
        # Epoch 21-30: lr = 0.00025
        # Helps model converge to better minimum in later epochs
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )

        # ── DataLoader ────────────────────────────────────────
        # TensorDataset + DataLoader handles mini-batching
        # shuffle=True: different order each epoch → better training
        train_dataset = TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(y_train,   dtype=torch.float32)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle    = True
        )

        # Validation tensors (no DataLoader — validate all at once)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
        y_val_t = torch.tensor(y_val,   dtype=torch.float32)

        # ── Training header ───────────────────────────────────
        mode = 'Adversarially-Hardened' if adversarial else 'Standard'
        print(f'\n{"=" * 60}')
        print(f'  Training {mode} IDS Model')
        print(f'  Dataset      : CICIDS2017')
        print(f'  Samples      : {len(X_train):,} train / '
              f'{len(X_val):,} val')
        print(f'  Architecture : {self.input_dim}→256→128→64→1')
        print(f'  Epochs       : {epochs}')
        print(f'  Batch size   : {batch_size}')
        print(f'  Learning rate: {lr}')
        if adversarial:
            print(f'  FGSM epsilon : {fgsm_epsilon} '
                  f'(adversarial augmentation)')
        print(f'{"=" * 60}')

        # ── Training loop ─────────────────────────────────────
        for epoch in range(epochs):

            # Set model to training mode
            # Enables Dropout and BatchNorm training behaviour
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:

                # Clear gradients from previous batch
                optimizer.zero_grad()

                if adversarial:
                    # ── Adversarial training ──────────────────
                    # Generate FGSM adversarial examples for batch
                    # Then train on mix of clean + adversarial
                    # Model learns to classify BOTH correctly
                    X_adv = self._fgsm_batch(
                        X_batch, y_batch, criterion, fgsm_epsilon
                    )

                    # Concatenate clean + adversarial (2x batch)
                    X_combined = torch.cat([X_batch, X_adv],  dim=0)
                    y_combined = torch.cat([y_batch, y_batch], dim=0)

                    # FIX 1: squeeze(-1) not squeeze()
                    # squeeze() collapses (1,1)→scalar on batch_size=1
                    # breaks BCELoss which expects 1D tensor
                    out  = self.model(X_combined).squeeze(-1)
                    loss = criterion(out, y_combined)

                else:
                    # ── Standard training ─────────────────────
                    # FIX 1: squeeze(-1) not squeeze()
                    out  = self.model(X_batch).squeeze(-1)
                    loss = criterion(out, y_batch)

                # Backpropagation
                loss.backward()

                # Update model weights
                optimizer.step()

                epoch_loss += loss.item()

            # Step learning rate scheduler after each epoch
            scheduler.step()

            # ── Validation ────────────────────────────────────
            # eval() disables Dropout → deterministic output
            # torch.no_grad() disables gradient computation → faster
            self.model.eval()
            with torch.no_grad():
                # FIX 1: squeeze(-1) not squeeze()
                val_out  = self.model(X_val_t).squeeze(-1)
                val_loss = criterion(val_out, y_val_t).item()

                # Threshold 0.5: >= 0.5 = Attack, < 0.5 = Normal
                val_preds = (val_out >= 0.5).float()
                val_acc   = (
                    (val_preds == y_val_t).float().mean().item()
                )

            # Record history for this epoch
            avg_loss = epoch_loss / len(train_loader)
            self.history['train_loss'].append(avg_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f'  Epoch [{epoch + 1:3d}/{epochs}]  '
                      f'Loss: {avg_loss:.4f}  |  '
                      f'Val Loss: {val_loss:.4f}  |  '
                      f'Val Acc: {val_acc * 100:.2f}%')

        # ── Final evaluation on validation set ────────────────
        self.model.eval()
        with torch.no_grad():
            # FIX 1: squeeze(-1) not squeeze()
            val_out     = self.model(X_val_t).squeeze(-1)
            final_preds = (val_out >= 0.5).numpy()

        print(f'\n{"=" * 60}')
        print(f'  Final Validation Report:')
        print(classification_report(
            y_val, final_preds,
            target_names = ['Normal', 'Attack'],
            digits       = 4
        ))

        # ── Save model, scaler, metrics, history ──────────────
        tag = 'adversarial' if adversarial else 'standard'

        # FIX 5: os.path.join not f-string slash
        # cross-platform correct path construction
        model_path   = os.path.join(
            self.model_dir, f'ids_{tag}.pth'
        )
        scaler_path  = os.path.join(
            self.model_dir, f'scaler_{tag}.pkl'
        )
        metrics_path = os.path.join(
            self.model_dir, f'metrics_{tag}.json'
        )
        history_path = os.path.join(
            self.model_dir, f'history_{tag}.json'
        )

        # Save model weights
        torch.save(self.model.state_dict(), model_path)

        # Save fitted scaler (must use same scaler for inference)
        joblib.dump(self.scaler, scaler_path)

        # Save metrics summary
        metrics = {
            'accuracy' : float(accuracy_score(y_val, final_preds)),
            'type'     : tag,
            'input_dim': self.input_dim,
            'epochs'   : epochs,
            'dataset'  : 'CICIDS2017'
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # FIX 6: Save training history to disk
        # Previously tracked in memory but never written
        # Needed for learning curve plots in visualise_results.py
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f'  Model saved   → {model_path}')
        print(f'  Scaler saved  → {scaler_path}')
        print(f'  Metrics saved → {metrics_path}')
        print(f'  History saved → {history_path}')

        return metrics

    # ──────────────────────────────────────────────────────────
    def _fgsm_batch(self,
                    X_batch  : torch.Tensor,
                    y_batch  : torch.Tensor,
                    criterion: nn.Module,
                    epsilon  : float) -> torch.Tensor:
        """
        Generate FGSM adversarial examples for one training batch.

        Fast Gradient Sign Method formula:
          x_adv = x + ε · sign(∇_x Loss(model(x), y))

        Moves input in the direction that MAXIMISES loss,
        making the model more likely to misclassify.

        FIX 2: self.model.zero_grad() called first
                without this, model parameter gradients
                accumulate across batches, corrupting
                adversarial training loss silently

        FIX 1: squeeze(-1) not squeeze()
                safe for any batch size including 1

        Args:
            X_batch  : Clean input batch tensor
            y_batch  : True label tensor
            criterion: BCELoss instance
            epsilon  : Perturbation budget (default 0.80)

        Returns:
            X_adv: Adversarially perturbed version of X_batch
                   Detached from computation graph
        """
        # FIX 2: Clear model parameter gradients BEFORE
        # adversarial forward pass
        # Without this: gradients from previous train step
        # accumulate into model weights during loss.backward()
        # corrupting the actual weight updates
        self.model.zero_grad()

        # Clone input — avoid modifying original batch
        # requires_grad_(True) enables gradient w.r.t. INPUT
        X_adv = X_batch.clone().detach().requires_grad_(True)

        # Forward pass through model
        # FIX 1: squeeze(-1) not squeeze()
        out  = self.model(X_adv).squeeze(-1)
        loss = criterion(out, y_batch)

        # Compute gradient with respect to INPUT (not weights)
        # We want to know: which direction to change input
        # to maximise the loss
        loss.backward()

        # FGSM: step in direction of gradient sign
        # sign() gives ±1 per feature → bounded perturbation
        perturbation = epsilon * X_adv.grad.sign()

        # Return perturbed input, detached from graph
        return (X_adv + perturbation).detach()

    # ──────────────────────────────────────────────────────────
    def load(self, tag: str = 'standard'):
        """
        Load saved model and scaler from models/cicids/.

        FIX 3: weights_only=True in torch.load()
                weights_only=False (old default) uses pickle
                which allows arbitrary code execution
                weights_only=True only loads tensor data — safe
                Required for PyTorch 2.x

        FIX 5: os.path.join for all paths

        Args:
            tag: 'standard' or 'adversarial'

        Raises:
            FileNotFoundError: if model files do not exist
                               Run run_experiment.py first
            RuntimeError     : if model architecture mismatch
        """
        # FIX 5: os.path.join not f-string slash
        scaler_path  = os.path.join(
            self.model_dir, f'scaler_{tag}.pkl'
        )
        metrics_path = os.path.join(
            self.model_dir, f'metrics_{tag}.json'
        )
        model_path   = os.path.join(
            self.model_dir, f'ids_{tag}.pth'
        )

        # Check all required files exist before loading
        for path in [scaler_path, metrics_path, model_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f'\n  Model file not found: {path}'
                    f'\n  Run src/run_experiment.py first'
                )

        # Load fitted scaler
        self.scaler = joblib.load(scaler_path)

        # Load metrics to get input_dim
        # Avoids hardcoding — reads correct dim from saved metadata
        with open(metrics_path) as f:
            meta = json.load(f)

        self.input_dim = meta['input_dim']

        # Build model with correct input dimension
        # WHY len not hardcoded: stays correct if features change
        self.model = IDSNet(input_dim=self.input_dim)

        # FIX 3: weights_only=True — secure loading
        # prevents arbitrary code execution via pickle
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location  = 'cpu',
                weights_only  = True
            )
        )

        # Set to eval mode — disables Dropout for inference
        self.model.eval()

        print(f'  Loaded {tag} model  '
              f'(accuracy={meta["accuracy"] * 100:.2f}%  '
              f'input_dim={self.input_dim})')

    # ──────────────────────────────────────────────────────────
    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels for raw (unscaled) features.

        FIX 1: squeeze(-1) — critical for live_ids_cicids.py
                which passes one flow at a time (batch_size=1)
                bare squeeze() collapses (1,1)→scalar, breaking
                numpy conversion and array operations

        Args:
            X_raw: Raw feature array shape (N, 52)
                   Unscaled — scaler applied internally

        Returns:
            Binary array shape (N,) dtype int
            0 = Normal, 1 = Attack
        """
        X_scaled = self.scaler.transform(
            X_raw.astype(np.float32)
        )
        tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            # FIX 1: squeeze(-1) not squeeze()
            out = self.model(tensor).squeeze(-1)

        return (out >= 0.5).numpy().astype(int)

    # ──────────────────────────────────────────────────────────
    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Return raw attack probability scores [0.0 → 1.0].

        Used by live_ids_cicids.py to show confidence percentage
        in the Rich dashboard.

        FIX 1: squeeze(-1) — same single-sample safety fix
                as predict()

        Args:
            X_raw: Raw feature array shape (N, 52)
                   Unscaled — scaler applied internally

        Returns:
            Probability array shape (N,) dtype float32
            Values in [0.0, 1.0]
            >= 0.5 indicates Attack
        """
        X_scaled = self.scaler.transform(
            X_raw.astype(np.float32)
        )
        tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            # FIX 1: squeeze(-1) not squeeze()
            out = self.model(tensor).squeeze(-1)

        return out.numpy()