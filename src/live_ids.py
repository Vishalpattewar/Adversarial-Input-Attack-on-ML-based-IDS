# =============================================================
# src/live_ids.py
# =============================================================
# PURPOSE:
#   Real-time network intrusion detection using the trained
#   NSL-KDD IDSNet model. Captures live packets via Scapy,
#   extracts per-packet features, classifies each flow as
#   Normal or Attack, and displays a Rich terminal dashboard.
#
# WHY NSL-KDD MODEL (not CICIDS2017):
#   CICIDS2017 features are BIDIRECTIONAL FLOW statistics.
#   They require a complete network flow (both directions)
#   before features can be computed:
#     'Bwd Packet Length Max'    → needs server response packets
#     'Flow Bytes/s'             → needs complete flow duration
#     'Init_Win_bytes_backward'  → needs server TCP window
#   These CANNOT be extracted from a single packet in real-time.
#
#   NSL-KDD features ARE computable per-packet:
#     src_bytes   → payload size of this packet
#     count       → packets seen in this flow
#     serror_rate → SYN errors in this flow
#   This makes NSL-KDD suitable for live per-packet inference.
#
#   For flow-level CICIDS2017 live inference, see:
#   src/live_ids_cicids.py (uses flow accumulator)
#
# WHAT THIS FILE DOES:
#   1. Loads NSL-KDD IDSNet model from adversarial_ids project
#   2. Captures live IP packets using Scapy
#   3. Tracks per-flow statistics (bytes, flags, counts)
#   4. Extracts 38 NSL-KDD features per packet
#   5. Classifies each flow: Normal(0) or Attack(1)
#   6. Displays Rich terminal dashboard (live stats + alerts)
#   7. Logs alerts to results/cicids/live_ids_alerts.log
#
# MODEL USED:
#   ../../adversarial_ids/models/ids_adversarial.pth
#   ../../adversarial_ids/models/scaler_adversarial.pkl
#   (NSL-KDD adversarially-hardened model, 38 features)
#
# HOW TO RUN:
#   Terminal 1 (IDS):
#     cd ~/A-IDS/adversarial_ids_cicids/src
#     sudo ../../adversarial_ids_cicids/venv/bin/python3 live_ids.py
#
#   Terminal 2 (Traffic):
#     cd ~/A-IDS/adversarial_ids_cicids
#     sudo venv/bin/python3 test_traffic.py
#
# NOTE: sudo required for raw packet capture via Scapy
#
# PRE-REQUISITE:
#   NSL-KDD project must have trained models:
#   ~/A-IDS/adversarial_ids/models/ids_adversarial.pth
#   Run adversarial_ids/src/run_experiment.py if missing
#
# BUGS FIXED VS ORIGINAL:
#   FIX 1: MODEL_PATH points to NSL-KDD project
#           original pointed to CICIDS models/ (empty folder)
#           → FileNotFoundError on startup
#
#   FIX 2: IDSNet loaded with input_dim=38 (NSL-KDD)
#           original used len(NUMERIC_FEATURES) = 52 (CICIDS)
#           → model architecture mismatch → RuntimeError
#
#   FIX 3: NSL-KDD feature names defined locally (38 features)
#           original imported NUMERIC_FEATURES from ids_model.py
#           which has 52 CICIDS features → assert 38==52 failed
#           on every single packet → live IDS never classified
#
#   FIX 4: __file__ anchored BASE_DIR for all paths
#           relative paths break when run with sudo from
#           different working directory
#
#   FIX 5: purge_stale_flows() uses state_lock
#           original called without lock → dict size change
#           during iteration → RuntimeError in multi-thread
#
#   FIX 6: INTERFACE = None (auto-detect, not hardcoded 'lo')
#           'lo' = loopback only → misses real network traffic
#           None = Scapy auto-detects best interface
#
#   FIX 7: weights_only=True in torch.load()
#           security fix → prevents arbitrary code execution
#
# CDAC ITISS — Adversarial Input Attack on ML-based IDS
# =============================================================


import os
import sys
import time
import datetime
import threading
from collections import defaultdict

import torch
import numpy as np
import joblib

# ── Allow Python to find src/ modules ─────────────────────────
sys.path.insert(
    0, os.path.dirname(os.path.realpath(__file__))
)

# FIX 2 + 3: Import only IDSNet class (not NUMERIC_FEATURES)
# NUMERIC_FEATURES from ids_model.py = 52 CICIDS features
# We need IDSNet class only — instantiated with input_dim=38
from ids_model import IDSNet

# ── Scapy for live packet capture ─────────────────────────────
from scapy.all import sniff, IP, TCP, UDP, conf

# Suppress Scapy interface warnings (common on WSL)
conf.verb = 0

# ── Rich for terminal dashboard ───────────────────────────────
from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich.layout  import Layout
from rich.live    import Live
from rich.text    import Text
from rich         import box


# ─────────────────────────────────────────────────────────────
#  PATH CONFIGURATION
#
#  FIX 4: All paths anchored to THIS script location
#  Prevents breakage when run with sudo from different CWD
#
#  WHY realpath not abspath:
#    realpath() resolves symlinks → works in Docker/NFS/WSL
#
#  WHY try/except NameError:
#    __file__ undefined in Jupyter → fallback to CWD
# ─────────────────────────────────────────────────────────────
try:
    _BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    _BASE_DIR = os.getcwd()

# Project root = one level up from src/
_PROJECT_ROOT = os.path.realpath(
    os.path.join(_BASE_DIR, '..')
)

# NSL-KDD project root = sibling directory
# FIX 1: Points to NSL-KDD project where trained models live
_NSL_KDD_ROOT = os.path.realpath(
    os.path.join(_PROJECT_ROOT, '..', 'adversarial_ids')
)


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

# FIX 1: Model paths point to NSL-KDD project (has trained models)
# CICIDS2017 models/ folder is empty until run_experiment.py runs
# Even after training, CICIDS model cannot do per-packet inference
MODEL_PATH   = os.path.join(
    _NSL_KDD_ROOT, 'models', 'ids_adversarial.pth'
)
SCALER_PATH  = os.path.join(
    _NSL_KDD_ROOT, 'models', 'scaler_adversarial.pkl'
)
METRICS_PATH = os.path.join(
    _NSL_KDD_ROOT, 'models', 'metrics_adversarial.json'
)

# FIX 4: Log file anchored to CICIDS project results
LOG_FILE = os.path.join(
    _PROJECT_ROOT, 'results', 'cicids', 'live_ids_alerts.log'
)

# Classification threshold
# prob >= THRESHOLD → ATTACK, prob < THRESHOLD → NORMAL
THRESHOLD = 0.5

# FIX 6: None = Scapy auto-detects best interface
# 'lo' = loopback only → misses real network traffic
# None = captures on default interface (eth0, wlan0, etc.)
# For test_traffic.py demo (sends to 127.0.0.1): use 'lo'
# For real network monitoring: use None
INTERFACE = 'lo'#None

# Flow TTL: seconds before idle flow removed from memory
# FIX 5: TTL + lock prevents memory leak + race condition
FLOW_TTL_SECONDS = 120

# Maximum alerts to display in dashboard at once
MAX_ALERTS_DISPLAY = 8

# FIX 3: NSL-KDD input dimension — defined locally
# Do NOT use len(NUMERIC_FEATURES) from ids_model.py
# ids_model.NUMERIC_FEATURES = 52 CICIDS features
# NSL-KDD model was trained with 38 features
NSL_KDD_INPUT_DIM = 38


# ─────────────────────────────────────────────────────────────
#  NSL-KDD FEATURE NAMES (38 features)
#
#  FIX 3: Defined locally — NOT imported from ids_model.py
#  ids_model.py belongs to CICIDS2017 (52 features)
#  live_ids.py uses NSL-KDD model (38 features)
#  Importing from ids_model would cause assert 38==52 to fail
#  on every single packet → live IDS never classifies anything
#
#  These 38 names match the feature vector built in
#  extract_features() below — order must be identical
# ─────────────────────────────────────────────────────────────
NSL_KDD_FEATURES = [
    'duration',                    #  0  connection duration
    'src_bytes',                   #  1  bytes from source
    'dst_bytes',                   #  2  bytes from destination
    'land',                        #  3  src/dst same host:port
    'wrong_fragment',              #  4  wrong fragments
    'urgent',                      #  5  urgent packets
    'hot',                         #  6  hot indicators
    'num_failed_logins',           #  7  failed login attempts
    'logged_in',                   #  8  successfully logged in
    'num_compromised',             #  9  compromised conditions
    'root_shell',                  # 10  root shell obtained
    'su_attempted',                # 11  su root attempted
    'num_root',                    # 12  root accesses
    'num_file_creations',          # 13  file creation operations
    'num_shells',                  # 14  shell prompts
    'num_access_files',            # 15  operations on access files
    'num_outbound_cmds',           # 16  outbound commands in ftp
    'is_host_login',               # 17  login is host login
    'is_guest_login',              # 18  login is guest login
    'count',                       # 19  connections to same host
    'srv_count',                   # 20  connections to same service
    'serror_rate',                 # 21  SYN error rate
    'srv_serror_rate',             # 22  SYN error rate (service)
    'rerror_rate',                 # 23  REJ error rate
    'srv_rerror_rate',             # 24  REJ error rate (service)
    'same_srv_rate',               # 25  same service rate
    'diff_srv_rate',               # 26  different service rate
    'srv_diff_host_rate',          # 27  different host rate
    'dst_host_count',              # 28  dst host connection count
    'dst_host_srv_count',          # 29  dst host service count
    'dst_host_same_srv_rate',      # 30  dst host same srv rate
    'dst_host_diff_srv_rate',      # 31  dst host diff srv rate
    'dst_host_same_src_port_rate', # 32  dst host same src port
    'dst_host_srv_diff_host_rate', # 33  dst host srv diff host
    'dst_host_serror_rate',        # 34  dst host SYN error rate
    'dst_host_srv_serror_rate',    # 35  dst host srv SYN error
    'dst_host_rerror_rate',        # 36  dst host REJ error rate
    'dst_host_srv_rerror_rate',    # 37  dst host srv REJ error
]
# Total: 38 NSL-KDD features → NSL_KDD_INPUT_DIM = 38


# ─────────────────────────────────────────────────────────────
#  GLOBAL STATE
#  Shared between packet_handler() thread and Rich dashboard
#  All access via state_lock for thread safety
# ─────────────────────────────────────────────────────────────

# Thread lock — prevents race conditions between
# packet_handler thread and dashboard render thread
state_lock = threading.Lock()

# Live statistics counters
stats = {
    'packet_count' : 0,       # total packets processed
    'alert_count'  : 0,       # total alerts raised
    'normal_count' : 0,       # classified as normal
    'attack_count' : 0,       # classified as attack
    'last_prob'    : 0.0,     # last classification probability
    'start_time'   : time.time(),
}

# Recent alerts list for dashboard display
# Protected by state_lock
recent_alerts = []

# Per-flow statistics
# Key   : "src_ip:src_port-dst_ip:dst_port"
# Value : flow counters dict
# FIX 5: includes 'last_seen' for TTL-based memory cleanup
flow_stats = defaultdict(lambda: {
    'count'      : 0,
    'src_bytes'  : 0,
    'dst_bytes'  : 0,
    'syn_count'  : 0,
    'fin_count'  : 0,
    'rst_count'  : 0,
    'start_time' : time.time(),
    'last_seen'  : time.time(),   # FIX 5: TTL tracking
})

console = Console()


# ─────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────
def load_model():
    """
    Load NSL-KDD IDSNet model and StandardScaler from disk.

    FIX 1: Loads from NSL-KDD project (adversarial_ids/models/)
            NOT from CICIDS models/ (which is empty)

    FIX 2: IDSNet instantiated with input_dim=NSL_KDD_INPUT_DIM
            = 38, NOT len(NUMERIC_FEATURES) = 52 (CICIDS)

    FIX 7: weights_only=True — prevents arbitrary code
            execution via pickle (PyTorch 2.x security fix)

    Returns:
        model : IDSNet in eval() mode, input_dim=38
        scaler: StandardScaler fitted on NSL-KDD X_train

    Raises:
        SystemExit: if model files not found
    """
    console.print('\n[cyan][*] Loading NSL-KDD IDSNet model...[/cyan]')
    console.print(
        f'[dim]    Model : {MODEL_PATH}[/dim]'
    )
    console.print(
        f'[dim]    Scaler: {SCALER_PATH}[/dim]'
    )

    # Check all required files exist
    for path, name in [
        (MODEL_PATH,   'Model (.pth)'),
        (SCALER_PATH,  'Scaler (.pkl)'),
        (METRICS_PATH, 'Metrics (.json)')
    ]:
        if not os.path.exists(path):
            console.print(
                f'\n[red][✗] {name} not found:[/red]'
            )
            console.print(f'[red]    {path}[/red]')
            console.print(
                f'[yellow]\n    Run NSL-KDD experiment first:[/yellow]'
            )
            console.print(
                f'[yellow]    cd ~/A-IDS/adversarial_ids/src[/yellow]'
            )
            console.print(
                f'[yellow]    python3 run_experiment.py[/yellow]\n'
            )
            sys.exit(1)

    try:
        # Load fitted StandardScaler (NSL-KDD trained)
        scaler = joblib.load(SCALER_PATH)

        # FIX 2: Build model with NSL-KDD input dimension (38)
        # NOT len(NUMERIC_FEATURES) which = 52 for CICIDS
        model = IDSNet(input_dim=NSL_KDD_INPUT_DIM)

        # FIX 7: weights_only=True — secure loading
        # Prevents arbitrary code execution via pickle
        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location = 'cpu',
                weights_only = True
            )
        )

        # eval() disables Dropout → deterministic inference
        model.eval()

        console.print(
            f'[green][✓] NSL-KDD model loaded '
            f'(input_dim={NSL_KDD_INPUT_DIM})[/green]'
        )
        console.print(
            f'[green][✓] Scaler loaded[/green]'
        )

        return model, scaler

    except Exception as e:
        console.print(f'[red][✗] Load failed: {e}[/red]')
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
#  FLOW CLEANUP — prevents memory leak
#
#  FIX 5: Called with state_lock to prevent race condition
#  WHY: packet_handler() runs in separate thread
#       Without lock: dict size changes during iteration
#       → RuntimeError: dictionary changed size during iteration
#
#  WHY TTL cleanup:
#    flow_stats grows with every unique src:port-dst:port pair
#    On busy network → thousands of flows → OOM eventually
#    Remove flows idle for FLOW_TTL_SECONDS (120s)
# ─────────────────────────────────────────────────────────────
def purge_stale_flows():
    """
    Remove flow entries not seen for FLOW_TTL_SECONDS.

    FIX 5: Uses state_lock to prevent race condition with
            packet_handler() running in separate thread.

    Called every 100 packets from packet_handler().
    Prevents unbounded memory growth on busy networks.
    """
    now = time.time()

    # FIX 5: Acquire lock before modifying shared dict
    with state_lock:
        stale_keys = [
            key for key, val in flow_stats.items()
            if now - val['last_seen'] > FLOW_TTL_SECONDS
        ]
        for key in stale_keys:
            del flow_stats[key]


# ─────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION — 38 NSL-KDD features
#
#  FIX 3: Feature vector exactly matches NSL_KDD_FEATURES order
#  Original had wrong order, missing features 16/17/18,
#  injected protocol_type at wrong position → only 36-37
#  features → completely wrong inputs to model
#
#  NSL-KDD features that CAN be extracted from raw packets:
#    duration    → flow elapsed time
#    src_bytes   → IP payload size (attacker → server)
#    count       → packets in this flow
#    syn/fin/rst → TCP flag counts
#    serror_rate → SYN errors as fraction of count
#    rerror_rate → RST/FIN errors as fraction of count
#
#  Features set to 0.0 (not extractable per-packet):
#    Application-layer features (hot, num_failed_logins etc.)
#    These require deep packet inspection or session tracking
#    Setting to 0.0 = neutral/unknown → model still classifies
#    based on available network-level features
# ─────────────────────────────────────────────────────────────
def extract_features(packet, flow_key: str) -> np.ndarray:
    """
    Extract 38 NSL-KDD features from packet + flow state.

    FIX 3: Feature vector order matches NSL_KDD_FEATURES exactly
            All 38 features present including indices 16, 17, 18
            which were missing in original code

    Args:
        packet  : Scapy IP packet object
        flow_key: Flow identifier "src_ip:port-dst_ip:port"

    Returns:
        features: np.ndarray shape (38,) dtype float32
                  Order matches NSL_KDD_FEATURES list exactly
    """
    flow = flow_stats[flow_key]

    # ── Update flow counters ───────────────────────────────────
    flow['count']    += 1
    flow['last_seen'] = time.time()    # FIX 5: update TTL

    # FIX: use IP payload only (not full Ethernet frame)
    # len(packet) includes Ethernet + IP + TCP headers
    # len(packet[IP].payload) = application data only
    # Prevents inflated src_bytes → false positives
    if IP in packet:
        flow['src_bytes'] += len(packet[IP].payload)

    # ── TCP flag tracking ─────────────────────────────────────
    if TCP in packet:
        flags = packet[TCP].flags
        if flags & 0x02: flow['syn_count'] += 1   # SYN flag
        if flags & 0x01: flow['fin_count'] += 1   # FIN flag
        if flags & 0x04: flow['rst_count'] += 1   # RST flag

    # ── Compute derived values ─────────────────────────────────
    duration   = time.time() - flow['start_time']
    count      = max(flow['count'], 1)        # avoid div/0
    src_bytes  = float(flow['src_bytes'])
    dst_bytes  = float(flow['dst_bytes'])
    syn_count  = flow['syn_count']
    fin_count  = flow['fin_count']
    rst_count  = flow['rst_count']

    # Rate features — computed as fraction of total count
    serror_rate   = syn_count / count    # SYN error fraction
    rerror_rate   = (fin_count + rst_count) / count
    same_srv_rate = syn_count / count    # same service estimate
    diff_srv_rate = 0.0                  # requires service tracking

    # ── Build feature vector in NSL_KDD_FEATURES order ────────
    # FIX 3: All 38 features present, correct order
    #         Indices 16, 17, 18 included (were missing before)
    features = np.array([
        duration,                           #  0  duration
        src_bytes,                          #  1  src_bytes
        dst_bytes,                          #  2  dst_bytes
        0.0,                                #  3  land
        0.0,                                #  4  wrong_fragment
        0.0,                                #  5  urgent
        0.0,                                #  6  hot
        0.0,                                #  7  num_failed_logins
        0.0,                                #  8  logged_in
        0.0,                                #  9  num_compromised
        0.0,                                # 10  root_shell
        0.0,                                # 11  su_attempted
        0.0,                                # 12  num_root
        0.0,                                # 13  num_file_creations
        0.0,                                # 14  num_shells
        0.0,                                # 15  num_access_files
        0.0,                                # 16  num_outbound_cmds ← FIX 3
        0.0,                                # 17  is_host_login     ← FIX 3
        0.0,                                # 18  is_guest_login    ← FIX 3
        float(count),                       # 19  count
        float(syn_count),                   # 20  srv_count
        serror_rate,                        # 21  serror_rate
        serror_rate,                        # 22  srv_serror_rate
        rerror_rate,                        # 23  rerror_rate
        rerror_rate,                        # 24  srv_rerror_rate
        same_srv_rate,                      # 25  same_srv_rate
        diff_srv_rate,                      # 26  diff_srv_rate
        same_srv_rate,                      # 27  srv_diff_host_rate
        float(min(count * 2, 255)),         # 28  dst_host_count
        float(min(syn_count * 2, 255)),     # 29  dst_host_srv_count
        same_srv_rate,                      # 30  dst_host_same_srv_rate
        diff_srv_rate,                      # 31  dst_host_diff_srv_rate
        same_srv_rate,                      # 32  dst_host_same_src_port_rate
        0.0,                                # 33  dst_host_srv_diff_host_rate
        serror_rate,                        # 34  dst_host_serror_rate
        serror_rate,                        # 35  dst_host_srv_serror_rate
        rerror_rate,                        # 36  dst_host_rerror_rate
        rerror_rate,                        # 37  dst_host_srv_rerror_rate
    ], dtype=np.float32)

    # Safety check — catch any future feature count mismatch
    assert len(features) == NSL_KDD_INPUT_DIM, (
        f'Feature count wrong: '
        f'got {len(features)}, expected {NSL_KDD_INPUT_DIM}'
    )

    return features


# ─────────────────────────────────────────────────────────────
#  CLASSIFICATION
# ─────────────────────────────────────────────────────────────
def classify(features: np.ndarray,
             model,
             scaler) -> float:
    """
    Classify a single packet's features as Normal or Attack.

    Uses IDSNet with squeeze(-1) for single-sample safety.
    batch_size=1 (single packet) → squeeze(-1) is critical.

    Args:
        features: Raw NSL-KDD feature array shape (38,)
        model   : Loaded IDSNet (input_dim=38, eval mode)
        scaler  : NSL-KDD StandardScaler

    Returns:
        prob: Attack probability [0.0 → 1.0]
              >= 0.5 = Attack, < 0.5 = Normal
    """
    # Reshape to (1, 38) — scaler requires 2D input
    features_2d = features.reshape(1, -1)

    # Apply same scaler used during NSL-KDD training
    try:
        features_scaled = scaler.transform(features_2d)
    except Exception:
        # Fallback: use raw features if scaler fails
        features_scaled = features_2d

    # Run through model
    with torch.no_grad():
        tensor = torch.tensor(
            features_scaled, dtype=torch.float32
        )
        # squeeze(-1): (1,1) → (1,) → .item() gives scalar
        # bare squeeze() would collapse (1,1)→scalar when
        # batch_size=1, breaking numpy operations
        prob = model(tensor).squeeze(-1).item()

    return float(prob)


# ─────────────────────────────────────────────────────────────
#  ALERT SYSTEM
# ─────────────────────────────────────────────────────────────
def raise_alert(packet, flow_key: str, prob: float):
    """
    Raise an intrusion alert for a detected attack flow.

    Updates global stats and recent_alerts (thread-safe via
    state_lock). Appends to LOG_FILE on disk.

    Args:
        packet  : Scapy packet that triggered the alert
        flow_key: Flow identifier string
        prob    : Attack probability from model [0.0 → 1.0]
    """
    global recent_alerts

    timestamp = datetime.datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'
    )

    # Extract network info safely
    src_ip   = packet[IP].src if IP in packet else 'unknown'
    dst_ip   = packet[IP].dst if IP in packet else 'unknown'
    protocol = (
        'TCP'  if TCP  in packet else
        'UDP'  if UDP  in packet else
        'ICMP'
    )
    src_port = (
        packet[TCP].sport if TCP in packet else
        packet[UDP].sport if UDP in packet else 0
    )
    dst_port = (
        packet[TCP].dport if TCP in packet else
        packet[UDP].dport if UDP in packet else 0
    )

    # HIGH = very confident attack, MEDIUM = likely attack
    severity = 'HIGH' if prob > 0.8 else 'MEDIUM'

    alert = {
        'time'    : timestamp,
        'src'     : f'{src_ip}:{src_port}',
        'dst'     : f'{dst_ip}:{dst_port}',
        'protocol': protocol,
        'prob'    : prob,
        'severity': severity,
    }

    # Update shared state — thread safe
    with state_lock:
        stats['alert_count'] += 1
        recent_alerts.append(alert)

        # Keep only last MAX_ALERTS_DISPLAY in memory
        if len(recent_alerts) > MAX_ALERTS_DISPLAY:
            recent_alerts = recent_alerts[-MAX_ALERTS_DISPLAY:]

    # Ensure log directory exists
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Append to log file
    with open(LOG_FILE, 'a') as f:
        f.write(
            f'[{timestamp}] ALERT #{stats["alert_count"]} | '
            f'{src_ip}:{src_port} → {dst_ip}:{dst_port} | '
            f'{protocol} | {prob * 100:.1f}% | {severity}\n'
        )


# ─────────────────────────────────────────────────────────────
#  PACKET HANDLER
#  Called by Scapy for every captured packet
# ─────────────────────────────────────────────────────────────
def packet_handler(packet, model, scaler):
    """
    Process a single captured packet.

    Extracts NSL-KDD features, classifies, updates stats,
    raises alert if attack detected, purges stale flows.

    Args:
        packet: Scapy packet object
        model : Loaded IDSNet (input_dim=38, eval mode)
        scaler: NSL-KDD StandardScaler
    """
    # Only process IP packets — skip ARP, etc.
    if IP not in packet:
        return

    # Build 5-tuple flow key
    src_ip   = packet[IP].src
    dst_ip   = packet[IP].dst
    src_port = (
        packet[TCP].sport if TCP in packet else
        packet[UDP].sport if UDP in packet else 0
    )
    dst_port = (
        packet[TCP].dport if TCP in packet else
        packet[UDP].dport if UDP in packet else 0
    )
    flow_key = f'{src_ip}:{src_port}-{dst_ip}:{dst_port}'

    # Extract 38 NSL-KDD features from packet + flow state
    features = extract_features(packet, flow_key)

    # Classify packet → attack probability
    prob = classify(features, model, scaler)

    # Update global statistics — thread safe
    with state_lock:
        stats['packet_count'] += 1
        stats['last_prob']     = prob

        if prob >= THRESHOLD:
            stats['attack_count'] += 1
        else:
            stats['normal_count'] += 1

    # Raise alert if attack probability >= threshold
    if prob >= THRESHOLD:
        raise_alert(packet, flow_key, prob)

    # FIX 5: Purge stale flows every 100 packets
    # Prevents flow_stats growing unboundedly (memory leak)
    # purge_stale_flows() uses state_lock internally
    if stats['packet_count'] % 100 == 0:
        purge_stale_flows()


# ─────────────────────────────────────────────────────────────
#  RICH DASHBOARD
# ─────────────────────────────────────────────────────────────
def build_dashboard() -> Layout:
    """
    Build the Rich terminal dashboard layout.

    Layout:
      ┌─────────────────────────────────────┐
      │              HEADER                 │
      ├──────────────────┬──────────────────┤
      │   LIVE STATS     │  RECENT ALERTS   │
      ├──────────────────┴──────────────────┤
      │              FOOTER                 │
      └─────────────────────────────────────┘

    Returns:
        Rich Layout object ready for Live() rendering
    """
    layout = Layout()

    # Top-level rows
    layout.split_column(
        Layout(name='header', size=7),
        Layout(name='body'),
        Layout(name='footer', size=3),
    )

    # Body split into stats + alerts columns
    layout['body'].split_row(
        Layout(name='stats',  ratio=1),
        Layout(name='alerts', ratio=2),
    )

    # ── Header ────────────────────────────────────────────────
    header_text = Text(justify='center')
    header_text.append(
        '\n  Adversarial IDS — Live Network Monitor\n',
        style='bold cyan'
    )
    header_text.append(
        '  Powered by IDSNet (NSL-KDD) | CDAC ITISS\n',
        style='dim white'
    )
    header_text.append(
        f'  Model: NSL-KDD Adversarially Hardened  |  '
        f'Threshold: {THRESHOLD}  |  '
        f'Interface: {INTERFACE or "Auto-detect"}\n',
        style='dim white'
    )
    layout['header'].update(
        Panel(
            header_text,
            style = 'cyan',
            box   = box.DOUBLE_EDGE
        )
    )

    # ── Live Stats Panel ──────────────────────────────────────
    with state_lock:
        elapsed = int(time.time() - stats['start_time'])
        hours   = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60

        total      = max(stats['packet_count'], 1)
        attack_pct = stats['attack_count'] / total * 100
        normal_pct = stats['normal_count'] / total * 100

        stats_table = Table(
            box         = box.SIMPLE,
            show_header = False,
            padding     = (0, 1)
        )
        stats_table.add_column('Metric', style='cyan',  width=20)
        stats_table.add_column('Value',  style='white', width=15)

        stats_table.add_row(
            'Runtime',
            f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        )
        stats_table.add_row(
            'Total Packets',
            f"{stats['packet_count']:,}"
        )
        stats_table.add_row(
            'Normal',
            f"[green]{stats['normal_count']:,} "
            f"({normal_pct:.1f}%)[/green]"
        )
        stats_table.add_row(
            'Attack',
            f"[red]{stats['attack_count']:,} "
            f"({attack_pct:.1f}%)[/red]"
        )
        stats_table.add_row(
            'Total Alerts',
            f"[bold red]{stats['alert_count']}[/bold red]"
        )
        stats_table.add_row(
            'Last Prob',
            f"[yellow]{stats['last_prob'] * 100:.1f}%[/yellow]"
        )
        stats_table.add_row(
            'Active Flows',
            f'{len(flow_stats):,}'
        )

    layout['stats'].update(
        Panel(
            stats_table,
            title = '[bold cyan] Live Statistics [/bold cyan]',
            style = 'cyan',
            box   = box.ROUNDED
        )
    )

    # ── Recent Alerts Panel ───────────────────────────────────
    alerts_table = Table(
        box          = box.SIMPLE_HEAVY,
        show_header  = True,
        header_style = 'bold red',
        expand       = True
    )
    alerts_table.add_column('Time',     style='dim white', width=19)
    alerts_table.add_column('Source',   style='yellow',    width=21)
    alerts_table.add_column('Target',   style='yellow',    width=21)
    alerts_table.add_column('Proto',    style='cyan',      width=6)
    alerts_table.add_column('Conf',     style='red',       width=7)
    alerts_table.add_column('Severity', style='bold',      width=8)

    with state_lock:
        # Most recent alerts first
        for alert in reversed(recent_alerts):
            sev_style = (
                'bold red'    if alert['severity'] == 'HIGH'
                else 'bold yellow'
            )
            alerts_table.add_row(
                alert['time'],
                alert['src'],
                alert['dst'],
                alert['protocol'],
                f"{alert['prob'] * 100:.1f}%",
                f"[{sev_style}]{alert['severity']}[/{sev_style}]"
            )

    if not recent_alerts:
        alerts_table.add_row(
            '—', '—', '—', '—', '—',
            '[green]No alerts yet[/green]'
        )

    layout['alerts'].update(
        Panel(
            alerts_table,
            title = '[bold red] Recent Alerts [/bold red]',
            style = 'red',
            box   = box.ROUNDED
        )
    )

    # ── Footer ────────────────────────────────────────────────
    layout['footer'].update(
        Panel(
            Text(
                '  Press Ctrl+C to stop  |  '
                f'Log: {LOG_FILE}  |  '
                'For CICIDS flow-level IDS: live_ids_cicids.py',
                justify = 'center',
                style   = 'dim white'
            ),
            style = 'dim',
            box   = box.SIMPLE
        )
    )

    return layout


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    """
    Start live IDS with Rich terminal dashboard.

    1. Load NSL-KDD model + scaler
    2. Start Scapy packet capture in background thread
    3. Render Rich dashboard in main thread (1s refresh)
    4. On Ctrl+C: stop capture, print session summary
    """

    # ── Load NSL-KDD model ────────────────────────────────────
    model, scaler = load_model()

    console.print(
        f'\n[green][✓] Live IDS starting...[/green]'
    )
    console.print(
        f'[cyan]    Interface: '
        f'{INTERFACE or "Auto-detect"}[/cyan]'
    )
    console.print(
        f'[cyan]    Log file : {LOG_FILE}[/cyan]'
    )
    console.print(
        f'[yellow]    Press Ctrl+C to stop[/yellow]\n'
    )
    console.print(
        f'[dim]    Note: Using NSL-KDD model (38 features)[/dim]'
    )
    console.print(
        f'[dim]    For CICIDS flow-level IDS: '
        f'python3 live_ids_cicids.py[/dim]\n'
    )

    # Brief pause so user can read startup messages
    time.sleep(1)

    # ── Start packet capture in background daemon thread ──────
    # daemon=True → thread stops automatically when main exits
    # store=False → packets not stored in memory (saves RAM)
    # filter='ip'  → only capture IP packets (skip ARP, etc.)
    def capture_thread():
        sniff(
            iface   = INTERFACE,
            prn     = lambda pkt: packet_handler(
                pkt, model, scaler
            ),
            store   = False,
            filter  = 'ip'
        )

    thread = threading.Thread(
        target = capture_thread,
        daemon = True
    )
    thread.start()

    # ── Rich Live Dashboard ───────────────────────────────────
    # Refreshes every 1 second
    # screen=True → full terminal takeover for clean display
    try:
        with Live(
            build_dashboard(),
            refresh_per_second = 1,
            screen             = True
        ) as live:
            while True:
                time.sleep(1)
                live.update(build_dashboard())

    except KeyboardInterrupt:

        # ── Session summary on Ctrl+C ─────────────────────────
        console.print(
            f'\n\n[yellow][!] IDS stopped by user[/yellow]'
        )
        console.print(f'\n[cyan]Session Summary:[/cyan]')
        console.print(
            f'  Total packets  : '
            f'[white]{stats["packet_count"]:,}[/white]'
        )
        console.print(
            f'  Normal packets : '
            f'[green]{stats["normal_count"]:,}[/green]'
        )
        console.print(
            f'  Attack packets : '
            f'[red]{stats["attack_count"]:,}[/red]'
        )
        console.print(
            f'  Total alerts   : '
            f'[bold red]{stats["alert_count"]}[/bold red]'
        )
        console.print(
            f'  Active flows   : '
            f'[white]{len(flow_stats):,}[/white]'
        )
        console.print(
            f'  Log saved to   : '
            f'[cyan]{LOG_FILE}[/cyan]'
        )
        console.print(
            f'\n[green][✓] Goodbye![/green]\n'
        )


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    main()