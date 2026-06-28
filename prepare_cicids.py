# =============================================================
# prepare_cicids.py
# =============================================================
# PURPOSE:
#   One-time preprocessing script that converts the cleaned
#   CICIDS2017 dataset into a sampled CSV file ready for
#   neural network training.
#
# INPUT:
#   data/cicids2017/cicids2017_cleaned.csv
#   → 685MB, 2,520,751 rows, 52 numeric features + 1 label
#   → Already cleaned: no inf, no whitespace, no duplicates
#   → Label column : 'Attack Type'
#   → Normal label : 'Normal Traffic'
#
# OUTPUT:
#   data/network_flows_cicids.csv
#   → 100,000 rows stratified sample
#   → 52 numeric features + binary label (0=Normal, 1=Attack)
#   → Ready for IDSTrainer in src/ids_model.py
#
# WHAT THIS FILE DOES (in order):
#   1. Anchors all paths to script location (not CWD)
#   2. Loads CSV with float32 dtype (saves ~1.4GB RAM)
#   3. Validates no NaN values exist
#   4. Validates no infinite values exist
#   5. Shows original class distribution
#   6. Converts labels to binary (Normal=0, Attack=1)
#   7. Selects 52 numeric features
#   8. Stratified sample of 100,000 rows
#   9. Validates output before saving
#  10. Saves to network_flows_cicids.csv
#
# KEY DESIGN DECISIONS:
#   WHY float32 on load:
#     Default float64 loads 685MB → ~2.8GB RAM
#     float32 loads 685MB → ~1.4GB RAM
#     Halves memory usage — important on 8GB WSL machine
#
#   WHY stratified sampling:
#     Maintains original 83.1% / 16.9% class ratio
#     Random sampling could accidentally change ratio
#     Ensures model trains on realistic distribution
#
#   WHY no StandardScaler here:
#     Scaler must be fitted on X_train ONLY
#     Fitting on full dataset = data leakage
#     Scaler is fitted inside IDSTrainer.train()
#
#   WHY __file__ anchored paths:
#     Relative paths break if script run from different directory
#     os.path.realpath(__file__) always finds script location
#     Works regardless of CWD
#
# HOW TO RUN (once only):
#   cd ~/A-IDS/adversarial_ids_cicids
#   source venv/bin/activate
#   python3 prepare_cicids.py
#
# NEXT STEP:
#   cd src && python3 run_experiment.py
#
# CDAC ITISS — Adversarial Input Attack on ML-based IDS
# =============================================================


import os
import sys
import time
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
#  PATH CONFIGURATION
#
#  WHY __file__ anchored:
#    If you run: python3 prepare_cicids.py         → works
#    If you run: python3 ../prepare_cicids.py       → works
#    If you run: python3 /full/path/prepare_cicids.py → works
#    Relative paths like 'data/...' only work from project root
#
#  WHY os.path.realpath (not abspath):
#    realpath() resolves symlinks → works in Docker/NFS/WSL
#    abspath() does not resolve symlinks
#
#  WHY try/except NameError:
#    __file__ is not defined in Jupyter notebooks
#    Fallback to os.getcwd() allows notebook usage
# ─────────────────────────────────────────────────────────────
try:
    # Anchor to directory containing this script
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    # Fallback for Jupyter notebook environment
    BASE_DIR = os.getcwd()
    print(f'  ⚠ __file__ not defined — using CWD: {BASE_DIR}')


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Input: single cleaned CICIDS2017 CSV (already cleaned)
INPUT_FILE = os.path.join(
    BASE_DIR, 'data', 'cicids2017', 'cicids2017_cleaned.csv'
)

# Output: stratified sample ready for training
OUTPUT_PATH = os.path.join(
    BASE_DIR, 'data', 'network_flows_cicids.csv'
)

# Label configuration
LABEL_COLUMN = 'Attack Type'    # column name in cicids2017_cleaned.csv
NORMAL_LABEL = 'Normal Traffic' # value that means legitimate traffic

# Sample size
# WHY 100,000:
#   Full 2.5M rows → training takes hours on CPU
#   100,000 rows   → training ~30-50 mins on CPU
#   Statistically sufficient for research results
#   Stratified to maintain 83.1% / 16.9% class ratio
N_SAMPLES = 100_000


# ─────────────────────────────────────────────────────────────
#  52 NUMERIC FEATURES
#
#  Exact column names from cicids2017_cleaned.csv
#  Order here = exact column order in output CSV
#  This list MUST match NUMERIC_FEATURES in src/ids_model.py
#
#  NOTE ON DESTINATION PORT (index 0):
#    Destination Port is technically metadata but kept because:
#    Port 80 flooding = likely DoS
#    Port 22 brute force = likely SSH attack
#    Already numeric → no encoding needed
#    Provides useful attack signal to the model
#
#  NOTE ON MISSING FEATURES vs ORIGINAL CICIDS2017:
#    Original had 76 features — this cleaned version has 52
#    24 features removed by dataset author because they were:
#    → Near-zero variance (e.g. Fwd Avg Bytes/Bulk)
#    → Redundant/correlated (e.g. Avg Fwd Segment Size)
#    → CICFlowMeter calculation errors (e.g. CWE Flag Count)
#    52 clean features > 76 noisy features for ML
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
# Total: 52 features → model input_dim = 52


# ─────────────────────────────────────────────────────────────
#  MAIN PREPROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────
def main():
    """
    Full preprocessing pipeline for CICIDS2017 dataset.

    Loads cleaned CSV, validates data quality, converts labels
    to binary, stratified samples 100,000 rows, saves output.

    No arguments — all configuration via constants above.
    """

    print('=' * 60)
    print('  CICIDS2017 Preprocessing')
    print('  Adversarial Input Attack on ML-based IDS')
    print('  CDAC ITISS')
    print('=' * 60)

    # ── Verify input file exists ──────────────────────────────
    # Hard stop if file missing — cannot proceed without data
    if not os.path.exists(INPUT_FILE):
        print(f'\n  ✗ ERROR: Input file not found:')
        print(f'    {INPUT_FILE}')
        print(f'\n  Download from Kaggle:')
        print(f'    kaggle datasets download → CICIDS2017 Cleaned')
        print(f'    Save to: data/cicids2017/cicids2017_cleaned.csv')
        print()
        sys.exit(1)

    # ── Step 1: Load CSV with float32 dtype ───────────────────
    # WHY float32: halves RAM usage (2.8GB → 1.4GB for 685MB CSV)
    # WHY low_memory=False: reads full file to infer types
    # WHY dtype dict: forces numeric cols to float32 immediately
    print(f'\n[1/7] Loading CSV file...')
    print(f'      File   : {INPUT_FILE}')
    print(f'      Size   : 685MB — may take 1-2 minutes')

    # Build dtype dict — all numeric features as float32
    # Label column stays as string for comparison
    dtype_dict = {feat: 'float32' for feat in NUMERIC_FEATURES}
    dtype_dict[LABEL_COLUMN] = 'str'

    t_start = time.time()
    df = pd.read_csv(INPUT_FILE, dtype=dtype_dict, low_memory=False)
    t_elapsed = time.time() - t_start

    print(f'      Loaded in {t_elapsed:.1f}s')
    print(f'      Rows    : {len(df):,}')
    print(f'      Columns : {len(df.columns)}')
    print(f'      RAM used: '
          f'{df.memory_usage(deep=True).sum() / 1e6:.0f} MB')

    # ── Step 2: Validate no NaN values ───────────────────────
    # WHY: Even "cleaned" datasets can have NaN from division
    #      by zero (e.g. Flow Bytes/s when duration=0)
    #      NaN propagates silently → model outputs NaN loss
    print(f'\n[2/7] Validating NaN values...')

    nan_counts  = df[NUMERIC_FEATURES].isnull().sum()
    nan_columns = nan_counts[nan_counts > 0]

    if len(nan_columns) > 0:
        print(f'      ⚠ NaN found in {len(nan_columns)} columns:')
        for col, count in nan_columns.items():
            print(f'        {col:<35} {count:,} NaNs')
        before = len(df)
        df     = df.dropna(subset=NUMERIC_FEATURES)
        print(f'      Dropped {before - len(df):,} rows with NaN')
    else:
        print(f'      No NaN values found ✓')

    # ── Step 3: Validate no infinite values ──────────────────
    # WHY: inf values cause gradient explosion during training
    #      StandardScaler fails on inf values
    print(f'\n[3/7] Validating infinite values...')

    inf_mask = np.isinf(df[NUMERIC_FEATURES]).any(axis=1)
    n_inf    = inf_mask.sum()

    if n_inf > 0:
        print(f'      ⚠ {n_inf:,} rows contain inf values')
        print(f'      Replacing inf with NaN then dropping...')
        df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].replace(
            [np.inf, -np.inf], np.nan
        )
        before = len(df)
        df     = df.dropna(subset=NUMERIC_FEATURES)
        print(f'      Dropped {before - len(df):,} rows with inf')
    else:
        print(f'      No infinite values found ✓')

    # ── Step 4: Show original label distribution ──────────────
    print(f'\n[4/7] Original label distribution:')
    label_counts = df[LABEL_COLUMN].value_counts()
    total_rows   = len(df)

    for label, count in label_counts.items():
        pct = count / total_rows * 100
        # Visual bar: 1 block per 2%
        bar = '█' * max(1, int(pct / 2))
        tag = '← normal' if label == NORMAL_LABEL else '← attack'
        print(f'      {label:<20} {count:>9,}  '
              f'({pct:5.1f}%)  {bar} {tag}')

    # ── Step 5: Convert labels to binary ─────────────────────
    # WHY binary (not multi-class):
    #   Project goal = detect IS it an attack (not which type)
    #   Binary is standard for IDS research benchmarks
    #   Simpler, faster, sufficient for adversarial analysis
    #
    # 'Normal Traffic' → 0  (legitimate)
    # All attack types → 1  (malicious)
    print(f'\n[5/7] Converting to binary labels...')
    print(f"      '{NORMAL_LABEL}' → 0  (legitimate)")
    print(f'      All attack types  → 1  (malicious)')

    df['label'] = (df[LABEL_COLUMN] != NORMAL_LABEL).astype(int)

    n_normal = (df['label'] == 0).sum()
    n_attack = (df['label'] == 1).sum()
    print(f'      Normal : {n_normal:,} ({n_normal/len(df)*100:.1f}%)')
    print(f'      Attack : {n_attack:,} ({n_attack/len(df)*100:.1f}%)')

    # ── Step 6: Stratified sampling ───────────────────────────
    # WHY stratified (not random):
    #   Maintains exact 83.1% / 16.9% class ratio from full dataset
    #   Random sampling could accidentally shift this ratio
    #   Ensures training data reflects real network distribution
    #
    # HOW: Sample from normal and attack separately,
    #      maintaining their proportion in full dataset
    print(f'\n[6/7] Stratified sampling to {N_SAMPLES:,} rows...')

    # Select only features + label columns
    df_numeric = df[NUMERIC_FEATURES + ['label']].copy()

    # Split by class
    normal_df = df_numeric[df_numeric['label'] == 0]
    attack_df = df_numeric[df_numeric['label'] == 1]

    # Calculate per-class sample sizes maintaining original ratio
    n_normal_sample = int(
        N_SAMPLES * len(normal_df) / len(df_numeric)
    )
    n_attack_sample = N_SAMPLES - n_normal_sample

    # Warn if requested sample exceeds available rows
    # (should not happen with 100K from 2.5M but defensive)
    if n_normal_sample > len(normal_df):
        print(f'      ⚠ Requested {n_normal_sample:,} normal — '
              f'only {len(normal_df):,} available, using all')
        n_normal_sample = len(normal_df)

    if n_attack_sample > len(attack_df):
        print(f'      ⚠ Requested {n_attack_sample:,} attack — '
              f'only {len(attack_df):,} available, using all')
        n_attack_sample = len(attack_df)

    print(f'      Normal sample : {n_normal_sample:,}')
    print(f'      Attack sample : {n_attack_sample:,}')

    # Sample each class separately
    # random_state=42 ensures reproducible results
    normal_sample = normal_df.sample(
        n=n_normal_sample, random_state=42
    )
    attack_sample = attack_df.sample(
        n=n_attack_sample, random_state=42
    )

    # Combine classes and shuffle
    # reset_index ensures clean 0-based index in output CSV
    df_final = pd.concat([normal_sample, attack_sample])
    df_final = df_final.sample(frac=1, random_state=42)
    df_final = df_final.reset_index(drop=True)

    # ── Step 7: Validate output before saving ─────────────────
    # WHY validate: catch any data issues before saving
    #   Wrong shape  → wrong features fed to model
    #   NaN in output → model training fails silently
    #   Wrong labels  → model learns wrong thing
    print(f'\n[7/7] Validating output...')

    # Check 1: Correct number of columns
    expected_cols = len(NUMERIC_FEATURES) + 1   # features + label
    assert df_final.shape[1] == expected_cols, (
        f'Wrong column count: '
        f'got {df_final.shape[1]}, expected {expected_cols}'
    )

    # Check 2: No NaN in output
    assert df_final.isnull().sum().sum() == 0, (
        f'NaN values in output: '
        f'{df_final.isnull().sum()[df_final.isnull().sum() > 0]}'
    )

    # Check 3: Labels are binary (only 0 and 1)
    assert set(df_final['label'].unique()).issubset({0, 1}), (
        f'Non-binary labels found: {df_final["label"].unique()}'
    )

    # Check 4: Correct number of features
    assert len(NUMERIC_FEATURES) == 52, (
        f'Expected 52 features, got {len(NUMERIC_FEATURES)}'
    )

    print(f'      Shape      : {df_final.shape} ✓')
    print(f'      NaN count  : 0 ✓')
    print(f'      Label vals : {sorted(df_final["label"].unique())} ✓')
    print(f'      Features   : {len(NUMERIC_FEATURES)} ✓')

    # ── Save to CSV ───────────────────────────────────────────
    # Ensure output directory exists (should exist but defensive)
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_final.to_csv(OUTPUT_PATH, index=False)

    # ── Final summary ─────────────────────────────────────────
    n_out_normal = (df_final['label'] == 0).sum()
    n_out_attack = (df_final['label'] == 1).sum()

    print(f'\n{"=" * 60}')
    print(f'  Preprocessing Complete')
    print(f'{"=" * 60}')
    print(f'  Output file  : {OUTPUT_PATH}')
    print(f'  Total rows   : {len(df_final):,}')
    print(f'  Normal rows  : {n_out_normal:,} '
          f'({n_out_normal / len(df_final) * 100:.1f}%)')
    print(f'  Attack rows  : {n_out_attack:,} '
          f'({n_out_attack / len(df_final) * 100:.1f}%)')
    print(f'  Features     : {len(NUMERIC_FEATURES)} numeric')
    print(f'  Label col    : label (0=Normal, 1=Attack)')
    print(f'  input_dim    : {len(NUMERIC_FEATURES)} (for IDSNet)')
    print(f'\n  Next step:')
    print(f'    cd src && python3 run_experiment.py')
    print(f'{"=" * 60}\n')


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    main()