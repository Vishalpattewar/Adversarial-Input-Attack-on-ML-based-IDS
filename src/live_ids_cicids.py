# =============================================================
# src/live_ids_cicids.py
# =============================================================
# PURPOSE:
#   Real-time network intrusion detection using the trained
#   CICIDS2017 IDSNet model. Implements a flow accumulator
#   that replicates the core logic of CICFlowMeter to extract
#   all 52 CICIDS2017 features from live bidirectional flows.
#
# HOW THIS DIFFERS FROM live_ids.py:
#   live_ids.py        → per-PACKET inference (NSL-KDD model)
#                         classifies every packet instantly
#                         38 features extractable per-packet
#
#   live_ids_cicids.py → per-FLOW inference (CICIDS2017 model)
#                         classifies after flow COMPLETES
#                         52 features require full flow data
#                         alert delay: 2-30 seconds per flow
#
# WHAT IS A FLOW ACCUMULATOR:
#   CICFlowMeter is a Java tool used to generate CICIDS2017.
#   It captures raw packets → tracks bidirectional flows →
#   computes 78 statistical features when a flow ends.
#
#   This file is a lightweight Python re-implementation:
#     Same concept: track flow → compute features → classify
#     Same features: all 52 from cicids2017_cleaned.csv
#     Difference  : Python (slower), 52 features (not 78)
#                   research demo (not production tool)
#
#   Academically this demonstrates:
#     ✅ Understanding of how CICIDS2017 was generated
#     ✅ Understanding of bidirectional flow statistics
#     ✅ Why CICIDS features need complete flows
#     ✅ CICIDS2017 model working on live traffic
#
# FLOW LIFECYCLE:
#   Packet arrives
#       ↓
#   Flow key built (src_ip:port → dst_ip:port)
#       ↓
#   Packet added to flow accumulator
#       ↓
#   Flow complete? (FIN+ACK / RST / TIMEOUT)
#       ↓ YES
#   Extract 52 CICIDS features from accumulated flow
#       ↓
#   Feed to IDSNet CICIDS model → probability
#       ↓
#   Alert if prob >= 0.5
#
# FLOW COMPLETION TRIGGERS:
#   1. TCP FIN + ACK  → clean connection close
#   2. TCP RST        → abrupt connection reset
#   3. Timeout        → no packet for FLOW_TIMEOUT seconds
#      (timeout flows classified on idle detection)
#
# THE 52 FEATURES — HOW EACH IS COMPUTED FROM LIVE FLOW:
#   Destination Port      → dst_port of first packet
#   Flow Duration         → last_time - first_time (μs)
#   Total Fwd Packets     → count of src→dst packets
#   Total Length Fwd      → sum of fwd payload bytes
#   Fwd Pkt Length Max    → max fwd payload size
#   Fwd Pkt Length Min    → min fwd payload size
#   Fwd Pkt Length Mean   → mean fwd payload size
#   Fwd Pkt Length Std    → std fwd payload size
#   Bwd Pkt Length Max    → max bwd payload size
#   ... (all 52 computed from accumulated statistics)
#   Active/Idle times     → burst detection with 1s threshold
#
# HOW TO RUN:
#   Terminal 1 (IDS):
#     cd ~/A-IDS/adversarial_ids_cicids/src
#     sudo ../../adversarial_ids_cicids/venv/bin/python3 \
#          live_ids_cicids.py
#
#   Terminal 2 (Traffic):
#     cd ~/A-IDS/adversarial_ids_cicids
#     sudo venv/bin/python3 test_traffic.py
#
# NOTE: sudo required for raw packet capture via Scapy
#
# PRE-REQUISITE:
#   run_experiment.py must be run first to train CICIDS model:
#   ~/A-IDS/adversarial_ids_cicids/models/cicids/
#       ids_adversarial.pth
#       scaler_adversarial.pkl
#
# CDAC ITISS — Adversarial Input Attack on ML-based IDS
# =============================================================


import os
import sys
import time
import datetime
import threading
import statistics

import torch
import numpy as np
import joblib

# ── Allow Python to find src/ modules ─────────────────────────
sys.path.insert(
    0, os.path.dirname(os.path.realpath(__file__))
)

# Import CICIDS2017 model components
# NUMERIC_FEATURES = 52 CICIDS features (correct for this file)
# IDSTrainer.load() handles model + scaler loading
from ids_model import IDSNet, NUMERIC_FEATURES, IDSTrainer

# ── Scapy for live packet capture ─────────────────────────────
from scapy.all import sniff, IP, TCP, UDP, conf

# Suppress Scapy interface warnings
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
#  All paths anchored to THIS script location
#  Prevents breakage when run with sudo from different CWD
# ─────────────────────────────────────────────────────────────
try:
    _BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    _BASE_DIR = os.getcwd()

# Project root = one level up from src/
_PROJECT_ROOT = os.path.realpath(
    os.path.join(_BASE_DIR, '..')
)


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

# CICIDS2017 model directory
# Populated after running run_experiment.py
MODELS_DIR = os.path.join(
    _PROJECT_ROOT, 'models', 'cicids'
)

# Log file for detected attacks
LOG_FILE = os.path.join(
    _PROJECT_ROOT, 'results', 'cicids',
    'live_ids_cicids_alerts.log'
)

# Classification threshold
# prob >= THRESHOLD → ATTACK, prob < THRESHOLD → NORMAL
THRESHOLD = 0.5

# Network interface to capture on
# None = auto-detect (all interfaces)
# 'lo' = loopback only (for test_traffic.py demo)
INTERFACE = 'lo' #None

# Flow timeout: seconds of inactivity before flow is closed
# and classified. Short flows (DoS, scan) will timeout quickly.
# Legitimate long flows (SSH, HTTP) timeout after idle period.
FLOW_TIMEOUT = 30

# Active/Idle burst detection threshold (seconds)
# Gap > this value between packets = new idle period starts
# Gap < this value = same active burst continues
BURST_THRESHOLD = 1.0

# Memory management
# Maximum flows to keep in memory simultaneously
MAX_FLOWS = 10_000

# Maximum alerts to display in dashboard
MAX_ALERTS_DISPLAY = 8

# How often to check for timed-out flows (seconds)
# Runs in background thread separate from packet capture
TIMEOUT_CHECK_INTERVAL = 5


# ─────────────────────────────────────────────────────────────
#  GLOBAL STATE
#  Shared between threads — all access via state_lock
# ─────────────────────────────────────────────────────────────

# Thread lock for all shared state
state_lock = threading.Lock()

# Global statistics counters
stats = {
    'packet_count' : 0,       # total packets captured
    'flow_count'   : 0,       # total flows completed + classified
    'alert_count'  : 0,       # total attack alerts raised
    'normal_count' : 0,       # flows classified as normal
    'attack_count' : 0,       # flows classified as attack
    'last_prob'    : 0.0,     # last classification probability
    'active_flows' : 0,       # flows currently being accumulated
    'start_time'   : time.time(),
}

# Recent alerts for dashboard display
recent_alerts = []

console = Console()


# ─────────────────────────────────────────────────────────────
#  FLOW ACCUMULATOR
#
#  Tracks bidirectional network flows.
#  Each flow identified by 5-tuple:
#    (src_ip, src_port, dst_ip, dst_port, protocol)
#
#  Forward direction  = src → dst (initiator → responder)
#  Backward direction = dst → src (responder → initiator)
#
#  This mirrors how CICFlowMeter tracks flows:
#    Fwd = packets from flow initiator
#    Bwd = packets from flow responder
#
#  Flow dict structure:
#    timing         → first_time, last_time, packet timestamps
#    fwd stats      → packet sizes, counts, IAT, header lengths
#    bwd stats      → packet sizes, counts, IAT, header lengths
#    flag counts    → FIN, PSH, ACK across all packets
#    TCP window     → initial window sizes (first packet each dir)
#    active/idle    → burst timing for Active/Idle features
#    meta           → dst_port, protocol, flow key
# ─────────────────────────────────────────────────────────────

# Active flows being accumulated
# Key   : flow_key string "src_ip:src_port-dst_ip:dst_port-proto"
# Value : flow dict (see _new_flow() below)
active_flows = {}


def _new_flow(dst_port: int, protocol: str) -> dict:
    """
    Create a new empty flow accumulator dict.

    Called when first packet of a new flow is seen.

    Args:
        dst_port : Destination port of this flow
        protocol : 'TCP', 'UDP', or 'ICMP'

    Returns:
        Empty flow dict ready for packet accumulation
    """
    now = time.time()
    return {
        # Timing
        'first_time'      : now,
        'last_time'        : now,
        'all_timestamps'  : [now],    # all packet arrival times

        # Forward direction (initiator → responder)
        'fwd_pkt_sizes'   : [],       # payload sizes per fwd pkt
        'fwd_timestamps'  : [now],    # arrival times of fwd pkts
        'fwd_header_len'  : 0,        # cumulative fwd header bytes
        'fwd_data_pkts'   : 0,        # fwd packets with payload>0

        # Backward direction (responder → initiator)
        'bwd_pkt_sizes'   : [],       # payload sizes per bwd pkt
        'bwd_timestamps'  : [],       # arrival times of bwd pkts
        'bwd_header_len'  : 0,        # cumulative bwd header bytes

        # TCP flags (all packets both directions)
        'fin_count'       : 0,
        'psh_count'       : 0,
        'ack_count'       : 0,
        'syn_count'       : 0,
        'rst_count'       : 0,
        'fin_ack_seen'    : False,    # clean close detected

        # TCP initial window sizes (first packet each direction)
        'init_win_fwd'    : -1,       # -1 = not yet seen
        'init_win_bwd'    : -1,

        # Minimum TCP segment size (fwd direction)
        'min_seg_fwd'     : float('inf'),

        # All packet sizes (both directions, for combined stats)
        'all_pkt_sizes'   : [],

        # Active / Idle burst tracking
        # Active = consecutive packets within BURST_THRESHOLD
        # Idle   = gap > BURST_THRESHOLD between packets
        'active_times'    : [],       # durations of active bursts
        'idle_times'      : [],       # durations of idle gaps
        'burst_start'     : now,      # start of current burst
        'last_pkt_time'   : now,      # time of previous packet

        # Metadata
        'dst_port'        : dst_port,
        'protocol'        : protocol,
    }


def _update_flow(flow        : dict,
                 is_fwd      : bool,
                 payload_size: int,
                 pkt_time    : float,
                 tcp_flags   : int,
                 win_size    : int,
                 header_len  : int,
                 total_len   : int):
    """
    Update flow accumulator with statistics from one packet.

    Called for every packet belonging to an existing flow.
    Separates forward and backward direction statistics.

    Args:
        flow        : Flow dict to update (modified in place)
        is_fwd      : True if packet is in forward direction
        payload_size: IP payload bytes in this packet
        pkt_time    : Packet arrival timestamp
        tcp_flags   : TCP flags integer (0 if not TCP)
        win_size    : TCP window size (0 if not TCP)
        header_len  : TCP/UDP header length in bytes
        total_len   : Total packet length (header + payload)
    """
    # ── Update timing ─────────────────────────────────────────
    flow['last_time'] = pkt_time
    flow['all_timestamps'].append(pkt_time)

    # ── Active / Idle burst detection ─────────────────────────
    # Gap since last packet determines active vs idle
    gap = pkt_time - flow['last_pkt_time']

    if gap > BURST_THRESHOLD:
        # Gap exceeds threshold → record idle period
        flow['idle_times'].append(gap)

        # Record active burst duration (time since burst started)
        active_dur = flow['last_pkt_time'] - flow['burst_start']
        if active_dur > 0:
            flow['active_times'].append(active_dur)

        # Start new burst
        flow['burst_start']  = pkt_time
    # else: same active burst continues

    flow['last_pkt_time'] = pkt_time

    # ── Update TCP flags ──────────────────────────────────────
    if tcp_flags:
        if tcp_flags & 0x01: flow['fin_count'] += 1  # FIN
        if tcp_flags & 0x02: flow['syn_count'] += 1  # SYN
        if tcp_flags & 0x04: flow['rst_count'] += 1  # RST
        if tcp_flags & 0x08: flow['psh_count'] += 1  # PSH
        if tcp_flags & 0x10: flow['ack_count'] += 1  # ACK

        # Detect clean connection close (FIN + ACK)
        if (tcp_flags & 0x01) and (tcp_flags & 0x10):
            flow['fin_ack_seen'] = True

    # ── All packet sizes ──────────────────────────────────────
    if total_len > 0:
        flow['all_pkt_sizes'].append(total_len)

    # ── Direction-specific updates ────────────────────────────
    if is_fwd:
        # Forward direction: initiator → responder
        flow['fwd_pkt_sizes'].append(payload_size)
        flow['fwd_timestamps'].append(pkt_time)
        flow['fwd_header_len'] += header_len

        # Track data packets (payload > 0)
        if payload_size > 0:
            flow['fwd_data_pkts'] += 1

        # Initial TCP window size (first fwd packet only)
        if flow['init_win_fwd'] == -1 and win_size > 0:
            flow['init_win_fwd'] = win_size

        # Minimum segment size (fwd TCP header length)
        if header_len > 0:
            flow['min_seg_fwd'] = min(
                flow['min_seg_fwd'], header_len
            )

    else:
        # Backward direction: responder → initiator
        flow['bwd_pkt_sizes'].append(payload_size)
        flow['bwd_timestamps'].append(pkt_time)
        flow['bwd_header_len'] += header_len

        # Initial TCP window size (first bwd packet only)
        if flow['init_win_bwd'] == -1 and win_size > 0:
            flow['init_win_bwd'] = win_size


def _compute_iat(timestamps: list) -> tuple:
    """
    Compute inter-arrival time statistics from timestamp list.

    Inter-arrival time (IAT) = time between consecutive packets
    in microseconds (to match CICFlowMeter units).

    Args:
        timestamps: Sorted list of packet arrival times (seconds)

    Returns:
        Tuple: (total, mean, std, max, min)
               All in microseconds
               Returns (0, 0, 0, 0, 0) if fewer than 2 packets
    """
    if len(timestamps) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Compute IAT in microseconds (×1,000,000)
    iats = [
        (timestamps[i] - timestamps[i-1]) * 1_000_000
        for i in range(1, len(timestamps))
    ]

    total = sum(iats)
    mean  = total / len(iats)
    std   = (
        statistics.stdev(iats)
        if len(iats) > 1 else 0.0
    )
    return total, mean, std, max(iats), min(iats)


def _safe_stats(values: list) -> tuple:
    """
    Compute max, min, mean, std safely on a list.

    Handles empty list and single-element list gracefully.

    Args:
        values: List of numeric values

    Returns:
        Tuple: (max, min, mean, std)
               Returns (0, 0, 0, 0) if list is empty
    """
    if not values:
        return 0.0, 0.0, 0.0, 0.0

    v_max  = max(values)
    v_min  = min(values)
    v_mean = sum(values) / len(values)
    v_std  = (
        statistics.stdev(values)
        if len(values) > 1 else 0.0
    )
    return v_max, v_min, v_mean, v_std


# ─────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION — 52 CICIDS2017 features
#
#  Called when a flow is completed (FIN/RST/timeout).
#  Computes all 52 features from accumulated flow statistics.
#  Feature order MUST match NUMERIC_FEATURES in ids_model.py
#
#  This is the core of the flow accumulator —
#  equivalent to what CICFlowMeter does when a flow ends.
# ─────────────────────────────────────────────────────────────
def extract_cicids_features(flow: dict) -> np.ndarray:
    """
    Extract 52 CICIDS2017 features from a completed flow.

    Computes the same statistical features as CICFlowMeter
    from the accumulated flow statistics dict.

    Feature order matches NUMERIC_FEATURES in ids_model.py.
    Any mismatch = wrong features fed to model = wrong output.

    Args:
        flow: Completed flow accumulator dict from _new_flow()
              Must have been updated via _update_flow()

    Returns:
        features: np.ndarray shape (52,) dtype float32
                  All values >= 0 (physical constraint)
    """
    # ── Flow duration (microseconds) ──────────────────────────
    duration_s  = max(
        flow['last_time'] - flow['first_time'], 1e-6
    )
    duration_us = duration_s * 1_000_000    # convert to μs

    # ── Packet size statistics ─────────────────────────────────
    fwd_sizes = flow['fwd_pkt_sizes'] or [0]
    bwd_sizes = flow['bwd_pkt_sizes'] or [0]
    all_sizes = flow['all_pkt_sizes'] or [0]

    fwd_max, fwd_min, fwd_mean, fwd_std = _safe_stats(fwd_sizes)
    bwd_max, bwd_min, bwd_mean, bwd_std = _safe_stats(bwd_sizes)
    all_max, all_min, all_mean, all_std = _safe_stats(all_sizes)

    # Packet length variance = std²
    pkt_variance = all_std ** 2

    # ── Packet counts ─────────────────────────────────────────
    n_fwd   = len(flow['fwd_pkt_sizes'])
    n_bwd   = len(flow['bwd_pkt_sizes'])
    n_total = max(n_fwd + n_bwd, 1)

    # ── Byte totals ───────────────────────────────────────────
    fwd_total_bytes = sum(fwd_sizes)
    bwd_total_bytes = sum(bwd_sizes)
    total_bytes     = fwd_total_bytes + bwd_total_bytes

    # ── Flow rates ────────────────────────────────────────────
    flow_bytes_s   = total_bytes / duration_s
    flow_pkts_s    = n_total     / duration_s
    fwd_pkts_s     = n_fwd       / duration_s
    bwd_pkts_s     = n_bwd       / duration_s

    # ── Inter-arrival times ───────────────────────────────────
    # All packets combined (both directions)
    all_ts_sorted = sorted(flow['all_timestamps'])
    (flow_iat_tot,
     flow_iat_mean,
     flow_iat_std,
     flow_iat_max,
     flow_iat_min) = _compute_iat(all_ts_sorted)

    # Forward direction only
    fwd_ts_sorted = sorted(flow['fwd_timestamps'])
    (fwd_iat_tot,
     fwd_iat_mean,
     fwd_iat_std,
     fwd_iat_max,
     fwd_iat_min) = _compute_iat(fwd_ts_sorted)

    # Backward direction only
    bwd_ts_sorted = sorted(flow['bwd_timestamps'])
    (bwd_iat_tot,
     bwd_iat_mean,
     bwd_iat_std,
     bwd_iat_max,
     bwd_iat_min) = _compute_iat(bwd_ts_sorted)

    # ── Header lengths ────────────────────────────────────────
    fwd_header = float(flow['fwd_header_len'])
    bwd_header = float(flow['bwd_header_len'])

    # ── Average packet size ───────────────────────────────────
    avg_pkt_size = total_bytes / n_total if n_total > 0 else 0.0

    # ── Subflow forward bytes ─────────────────────────────────
    # Subflow = same as full flow in CICFlowMeter
    # (no subflow division implemented in accumulator)
    subflow_fwd_bytes = float(fwd_total_bytes)

    # ── TCP window sizes ──────────────────────────────────────
    init_win_fwd = float(
        max(flow['init_win_fwd'], 0)
    )
    init_win_bwd = float(
        max(flow['init_win_bwd'], 0)
    )

    # ── Active / Idle times ───────────────────────────────────
    # Record final active burst before extracting
    final_active = flow['last_time'] - flow['burst_start']
    active_list  = flow['active_times'].copy()
    if final_active > 0:
        active_list.append(final_active)
    idle_list = flow['idle_times']

    (act_max, act_min,
     act_mean, _)    = _safe_stats(active_list)
    (idle_max, idle_min,
     idle_mean, _)   = _safe_stats(idle_list)

    # Convert active/idle to microseconds
    act_mean_us  = act_mean  * 1_000_000
    act_max_us   = act_max   * 1_000_000
    act_min_us   = act_min   * 1_000_000
    idle_mean_us = idle_mean * 1_000_000
    idle_max_us  = idle_max  * 1_000_000
    idle_min_us  = idle_min  * 1_000_000

    # ── min_seg_size_forward ──────────────────────────────────
    min_seg_fwd = (
        float(flow['min_seg_fwd'])
        if flow['min_seg_fwd'] != float('inf')
        else 0.0
    )

    # ── Build feature vector ──────────────────────────────────
    # Order MUST match NUMERIC_FEATURES in ids_model.py
    features = np.array([
        float(flow['dst_port']),     #  0  Destination Port
        duration_us,                  #  1  Flow Duration (μs)
        float(n_fwd),                 #  2  Total Fwd Packets
        float(fwd_total_bytes),       #  3  Total Length Fwd Pkts
        fwd_max,                      #  4  Fwd Pkt Length Max
        fwd_min,                      #  5  Fwd Pkt Length Min
        fwd_mean,                     #  6  Fwd Pkt Length Mean
        fwd_std,                      #  7  Fwd Pkt Length Std
        bwd_max,                      #  8  Bwd Pkt Length Max
        bwd_min,                      #  9  Bwd Pkt Length Min
        bwd_mean,                     # 10  Bwd Pkt Length Mean
        bwd_std,                      # 11  Bwd Pkt Length Std
        flow_bytes_s,                 # 12  Flow Bytes/s
        flow_pkts_s,                  # 13  Flow Packets/s
        flow_iat_mean,                # 14  Flow IAT Mean
        flow_iat_std,                 # 15  Flow IAT Std
        flow_iat_max,                 # 16  Flow IAT Max
        flow_iat_min,                 # 17  Flow IAT Min
        fwd_iat_tot,                  # 18  Fwd IAT Total
        fwd_iat_mean,                 # 19  Fwd IAT Mean
        fwd_iat_std,                  # 20  Fwd IAT Std
        fwd_iat_max,                  # 21  Fwd IAT Max
        fwd_iat_min,                  # 22  Fwd IAT Min
        bwd_iat_tot,                  # 23  Bwd IAT Total
        bwd_iat_mean,                 # 24  Bwd IAT Mean
        bwd_iat_std,                  # 25  Bwd IAT Std
        bwd_iat_max,                  # 26  Bwd IAT Max
        bwd_iat_min,                  # 27  Bwd IAT Min
        fwd_header,                   # 28  Fwd Header Length
        bwd_header,                   # 29  Bwd Header Length
        fwd_pkts_s,                   # 30  Fwd Packets/s
        bwd_pkts_s,                   # 31  Bwd Packets/s
        float(all_min),               # 32  Min Packet Length
        float(all_max),               # 33  Max Packet Length
        all_mean,                     # 34  Packet Length Mean
        all_std,                      # 35  Packet Length Std
        pkt_variance,                 # 36  Packet Length Variance
        float(flow['fin_count']),     # 37  FIN Flag Count
        float(flow['psh_count']),     # 38  PSH Flag Count
        float(flow['ack_count']),     # 39  ACK Flag Count
        avg_pkt_size,                 # 40  Average Packet Size
        subflow_fwd_bytes,            # 41  Subflow Fwd Bytes
        init_win_fwd,                 # 42  Init_Win_bytes_forward
        init_win_bwd,                 # 43  Init_Win_bytes_backward
        float(flow['fwd_data_pkts']), # 44  act_data_pkt_fwd
        min_seg_fwd,                  # 45  min_seg_size_forward
        act_mean_us,                  # 46  Active Mean
        act_max_us,                   # 47  Active Max
        act_min_us,                   # 48  Active Min
        idle_mean_us,                 # 49  Idle Mean
        idle_max_us,                  # 50  Idle Max
        idle_min_us,                  # 51  Idle Min
    ], dtype=np.float32)

    # Clip all features to >= 0 (physical constraint)
    # No negative packet counts, sizes, rates, or times
    features = np.clip(features, a_min=0.0, a_max=None)

    # Safety check — catch feature count mismatch immediately
    assert len(features) == len(NUMERIC_FEATURES), (
        f'CICIDS feature count wrong: '
        f'got {len(features)}, '
        f'expected {len(NUMERIC_FEATURES)}'
    )

    return features


# ─────────────────────────────────────────────────────────────
#  FLOW CLASSIFICATION
# ─────────────────────────────────────────────────────────────
def classify_flow(flow   : dict,
                  trainer: IDSTrainer) -> float:
    """
    Classify a completed flow using CICIDS2017 IDSNet model.

    Extracts 52 features → scales via StandardScaler →
    feeds to IDSNet → returns attack probability.

    Args:
        flow   : Completed flow dict from active_flows
        trainer: Loaded IDSTrainer with CICIDS2017 model

    Returns:
        prob: Attack probability [0.0 → 1.0]
              >= 0.5 = Attack, < 0.5 = Normal
    """
    # Extract 52 CICIDS features from flow accumulator
    features = extract_cicids_features(flow)

    # Reshape to (1, 52) for scaler + model
    features_2d = features.reshape(1, -1)

    # Use IDSTrainer.predict_proba() which handles:
    # → scaler.transform() (same scaler as training)
    # → tensor conversion
    # → model forward pass with squeeze(-1)
    proba = trainer.predict_proba(features_2d)

    # proba shape (1,) → scalar
    return float(proba[0])


# ─────────────────────────────────────────────────────────────
#  FLOW COMPLETION + ALERT
# ─────────────────────────────────────────────────────────────
def complete_flow(flow_key: str,
                  trainer : IDSTrainer,
                  reason  : str):
    """
    Finalise a flow, classify it, raise alert if attack.

    Called when:
      - TCP FIN+ACK seen (clean close)
      - TCP RST seen     (abrupt close)
      - Flow timeout     (no packets for FLOW_TIMEOUT seconds)

    Args:
        flow_key: Key in active_flows dict
        trainer : Loaded IDSTrainer (CICIDS2017 model)
        reason  : Why flow completed ('FIN', 'RST', 'TIMEOUT')
    """
    with state_lock:
        flow = active_flows.pop(flow_key, None)
        if flow is None:
            return  # already removed by another thread
        stats['active_flows'] = len(active_flows)

    # Need at least 2 packets (fwd) to classify meaningfully
    # Single-packet flows have no IAT, rate stats → unreliable
    if len(flow['fwd_pkt_sizes']) < 2:
        return

    # Classify flow using CICIDS2017 model
    try:
        prob = classify_flow(flow, trainer)
    except Exception:
        # Classification failed — skip this flow
        return

    # Update global stats
    with state_lock:
        stats['flow_count'] += 1
        stats['last_prob']   = prob

        if prob >= THRESHOLD:
            stats['attack_count'] += 1
        else:
            stats['normal_count'] += 1

    # Raise alert if attack detected
    if prob >= THRESHOLD:
        _raise_flow_alert(flow, flow_key, prob, reason)


def _raise_flow_alert(flow    : dict,
                      flow_key: str,
                      prob    : float,
                      reason  : str):
    """
    Raise and log an alert for a classified attack flow.

    Updates recent_alerts list and writes to LOG_FILE.

    Args:
        flow    : Completed flow dict
        flow_key: Flow identifier string
        prob    : Attack probability from CICIDS model
        reason  : Flow completion reason ('FIN','RST','TIMEOUT')
    """
    global recent_alerts

    timestamp = datetime.datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'
    )

    # Parse flow key: "src_ip:src_port-dst_ip:dst_port-proto"
    try:
        parts    = flow_key.split('-')
        src_part = parts[0]
        dst_part = parts[1]
        protocol = parts[2] if len(parts) > 2 else 'UNK'
    except (IndexError, ValueError):
        src_part = flow_key
        dst_part = 'unknown'
        protocol = 'UNK'

    severity = 'HIGH' if prob > 0.8 else 'MEDIUM'

    # Flow duration in milliseconds for display
    duration_ms = (
        flow['last_time'] - flow['first_time']
    ) * 1000

    alert = {
        'time'       : timestamp,
        'src'        : src_part,
        'dst'        : dst_part,
        'protocol'   : protocol,
        'prob'       : prob,
        'severity'   : severity,
        'reason'     : reason,
        'duration_ms': round(duration_ms, 1),
        'pkts'       : (
            len(flow['fwd_pkt_sizes']) +
            len(flow['bwd_pkt_sizes'])
        )
    }

    # Update shared state — thread safe
    with state_lock:
        stats['alert_count'] += 1
        recent_alerts.append(alert)

        # Keep only last MAX_ALERTS_DISPLAY
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
            f'{src_part} → {dst_part} | '
            f'{protocol} | '
            f'{prob * 100:.1f}% | '
            f'{severity} | '
            f'{reason} | '
            f'{duration_ms:.0f}ms\n'
        )


# ─────────────────────────────────────────────────────────────
#  PACKET HANDLER
#  Called by Scapy for every captured packet
# ─────────────────────────────────────────────────────────────
def packet_handler(packet, trainer: IDSTrainer):
    """
    Process a single captured packet into the flow accumulator.

    1. Build flow key from 5-tuple
    2. Create new flow or update existing flow
    3. Check for flow completion (FIN/RST)
    4. Update global packet counter

    Args:
        packet : Scapy IP packet object
        trainer: Loaded IDSTrainer (CICIDS2017 model)
    """
    # Only process IP packets
    if IP not in packet:
        return

    pkt_time = time.time()

    # ── Extract packet fields ─────────────────────────────────
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst

    if TCP in packet:
        src_port    = packet[TCP].sport
        dst_port    = packet[TCP].dport
        tcp_flags   = int(packet[TCP].flags)
        win_size    = packet[TCP].window
        # TCP header length = data offset × 4 bytes
        header_len  = packet[TCP].dataofs * 4
        protocol    = 'TCP'
    elif UDP in packet:
        src_port    = packet[UDP].sport
        dst_port    = packet[UDP].dport
        tcp_flags   = 0
        win_size    = 0
        header_len  = 8   # UDP header is always 8 bytes
        protocol    = 'UDP'
    else:
        # ICMP or other — use port 0
        src_port    = 0
        dst_port    = 0
        tcp_flags   = 0
        win_size    = 0
        header_len  = 0
        protocol    = 'ICMP'

    # IP payload size = application data bytes
    payload_size = len(packet[IP].payload)

    # Total packet length including headers
    total_len = len(packet)

    # ── Build canonical flow key ──────────────────────────────
    # Canonical = lower IP first to group both directions
    # Forward  = src → dst (as packet arrived)
    # Backward = dst → src
    if src_ip <= dst_ip:
        flow_key = (
            f'{src_ip}:{src_port}'
            f'-{dst_ip}:{dst_port}'
            f'-{protocol}'
        )
        is_fwd = True
    else:
        flow_key = (
            f'{dst_ip}:{dst_port}'
            f'-{src_ip}:{src_port}'
            f'-{protocol}'
        )
        is_fwd = False

    # ── Update global packet count ────────────────────────────
    with state_lock:
        stats['packet_count'] += 1

    # ── Update flow accumulator ───────────────────────────────
    with state_lock:
        if flow_key not in active_flows:
            # New flow — initialise accumulator
            if len(active_flows) >= MAX_FLOWS:
                # Memory protection: evict oldest flow
                oldest_key = min(
                    active_flows,
                    key=lambda k: active_flows[k]['first_time']
                )
                active_flows.pop(oldest_key, None)

            active_flows[flow_key] = _new_flow(
                dst_port, protocol
            )

        # Update flow with this packet's data
        _update_flow(
            flow         = active_flows[flow_key],
            is_fwd       = is_fwd,
            payload_size = payload_size,
            pkt_time     = pkt_time,
            tcp_flags    = tcp_flags,
            win_size     = win_size,
            header_len   = header_len,
            total_len    = total_len
        )

        stats['active_flows'] = len(active_flows)

    # ── Check for TCP flow completion ─────────────────────────
    # Complete flow outside lock to avoid blocking capture thread
    with state_lock:
        flow_exists = flow_key in active_flows
        if flow_exists:
            fin_ack = active_flows[flow_key]['fin_ack_seen']
            rst     = bool(tcp_flags & 0x04)
        else:
            fin_ack = False
            rst     = False

    if flow_exists:
        if fin_ack:
            # Clean TCP close — classify flow
            complete_flow(flow_key, trainer, 'FIN')
        elif rst:
            # Abrupt reset — classify flow
            complete_flow(flow_key, trainer, 'RST')


# ─────────────────────────────────────────────────────────────
#  TIMEOUT MONITOR
#  Background thread that classifies idle flows
# ─────────────────────────────────────────────────────────────
def timeout_monitor(trainer: IDSTrainer):
    """
    Background thread that classifies flows idle for too long.

    Runs every TIMEOUT_CHECK_INTERVAL seconds.
    Finds flows with no packet for FLOW_TIMEOUT seconds.
    Classifies and removes them.

    WHY THIS EXISTS:
      Many attacks (DoS, scan) are short-lived.
      They end without FIN/RST → never complete via TCP close.
      Timeout ensures they get classified eventually.

    Args:
        trainer: Loaded IDSTrainer (CICIDS2017 model)
    """
    while True:
        time.sleep(TIMEOUT_CHECK_INTERVAL)

        now     = time.time()
        timed_out = []

        # Find timed-out flows (safe read under lock)
        with state_lock:
            for key, flow in active_flows.items():
                if now - flow['last_time'] > FLOW_TIMEOUT:
                    timed_out.append(key)

        # Complete each timed-out flow
        for key in timed_out:
            complete_flow(key, trainer, 'TIMEOUT')


# ─────────────────────────────────────────────────────────────
#  RICH DASHBOARD
# ─────────────────────────────────────────────────────────────
def build_dashboard() -> Layout:
    """
    Build Rich terminal dashboard for CICIDS2017 flow IDS.

    Layout:
      ┌──────────────────────────────────────┐
      │              HEADER                  │
      ├──────────────────┬───────────────────┤
      │   LIVE STATS     │   RECENT ALERTS   │
      ├──────────────────┴───────────────────┤
      │              FOOTER                  │
      └──────────────────────────────────────┘

    Returns:
        Rich Layout object ready for Live() rendering
    """
    layout = Layout()

    layout.split_column(
        Layout(name='header', size=7),
        Layout(name='body'),
        Layout(name='footer', size=3),
    )
    layout['body'].split_row(
        Layout(name='stats',  ratio=1),
        Layout(name='alerts', ratio=2),
    )

    # ── Header ────────────────────────────────────────────────
    header_text = Text(justify='center')
    header_text.append(
        '\n  Adversarial IDS — CICIDS2017 Flow Monitor\n',
        style='bold cyan'
    )
    header_text.append(
        '  Flow Accumulator + IDSNet (52 features) | '
        'CDAC ITISS\n',
        style='dim white'
    )
    header_text.append(
        f'  Model: CICIDS2017 Adversarially Hardened  |  '
        f'Threshold: {THRESHOLD}  |  '
        f'Flow Timeout: {FLOW_TIMEOUT}s\n',
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

        total_flows = max(stats['flow_count'], 1)
        attack_pct  = stats['attack_count'] / total_flows * 100
        normal_pct  = stats['normal_count'] / total_flows * 100

        stats_table = Table(
            box         = box.SIMPLE,
            show_header = False,
            padding     = (0, 1)
        )
        stats_table.add_column(
            'Metric', style='cyan',  width=22
        )
        stats_table.add_column(
            'Value',  style='white', width=15
        )

        stats_table.add_row(
            'Runtime',
            f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        )
        stats_table.add_row(
            'Packets Captured',
            f"{stats['packet_count']:,}"
        )
        stats_table.add_row(
            'Flows Classified',
            f"{stats['flow_count']:,}"
        )
        stats_table.add_row(
            'Active Flows',
            f"[yellow]{stats['active_flows']:,}[/yellow]"
        )
        stats_table.add_row(
            'Normal Flows',
            f"[green]{stats['normal_count']:,} "
            f"({normal_pct:.1f}%)[/green]"
        )
        stats_table.add_row(
            'Attack Flows',
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

    layout['stats'].update(
        Panel(
            stats_table,
            title = '[bold cyan] Flow Statistics [/bold cyan]',
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
    alerts_table.add_column('Source',   style='yellow',    width=20)
    alerts_table.add_column('Proto',    style='cyan',      width=5)
    alerts_table.add_column('Pkts',     style='white',     width=5)
    alerts_table.add_column('Conf',     style='red',       width=7)
    alerts_table.add_column('Trigger',  style='magenta',   width=8)
    alerts_table.add_column('Severity', style='bold',      width=8)

    with state_lock:
        for alert in reversed(recent_alerts):
            sev_style = (
                'bold red'
                if alert['severity'] == 'HIGH'
                else 'bold yellow'
            )
            alerts_table.add_row(
                alert['time'],
                alert['src'],
                alert['protocol'],
                str(alert['pkts']),
                f"{alert['prob'] * 100:.1f}%",
                alert['reason'],
                f"[{sev_style}]"
                f"{alert['severity']}"
                f"[/{sev_style}]"
            )

    if not recent_alerts:
        alerts_table.add_row(
            '—', '—', '—', '—', '—', '—',
            '[green]No alerts yet[/green]'
        )

    layout['alerts'].update(
        Panel(
            alerts_table,
            title = '[bold red] Flow Alerts [/bold red]',
            style = 'red',
            box   = box.ROUNDED
        )
    )

    # ── Footer ────────────────────────────────────────────────
    layout['footer'].update(
        Panel(
            Text(
                f'  Press Ctrl+C to stop  |  '
                f'Log: {LOG_FILE}  |  '
                f'Flow timeout: {FLOW_TIMEOUT}s  |  '
                f'Burst threshold: {BURST_THRESHOLD}s',
                justify = 'center',
                style   = 'dim white'
            ),
            style = 'dim',
            box   = box.SIMPLE
        )
    )

    return layout


# ─────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────
def load_cicids_model() -> IDSTrainer:
    """
    Load CICIDS2017 IDSNet model using IDSTrainer.load().

    Uses adversarial model (hardened — more robust).
    Model directory: models/cicids/

    Returns:
        trainer: IDSTrainer with loaded model + scaler
                 Ready for predict_proba() calls

    Raises:
        SystemExit: if model files not found
    """
    console.print(
        '\n[cyan][*] Loading CICIDS2017 IDSNet model...[/cyan]'
    )
    console.print(
        f'[dim]    Directory: {MODELS_DIR}[/dim]'
    )

    # Check models directory exists
    if not os.path.exists(MODELS_DIR):
        console.print(
            f'\n[red][✗] Models directory not found:[/red]'
        )
        console.print(f'[red]    {MODELS_DIR}[/red]')
        console.print(
            f'[yellow]\n    Run run_experiment.py first:[/yellow]'
        )
        console.print(
            f'[yellow]    cd ~/A-IDS/adversarial_ids_cicids/src'
            f'[/yellow]'
        )
        console.print(
            f'[yellow]    python3 run_experiment.py[/yellow]\n'
        )
        sys.exit(1)

    try:
        # IDSTrainer.load() handles:
        # → loads scaler_{tag}.pkl
        # → reads input_dim from metrics_{tag}.json
        # → builds IDSNet with correct input_dim
        # → loads ids_{tag}.pth with weights_only=True
        # → sets model.eval()
        trainer = IDSTrainer(model_dir=MODELS_DIR)
        trainer.load(tag='adversarial')

        console.print(
            f'[green][✓] CICIDS2017 model loaded '
            f'(input_dim={len(NUMERIC_FEATURES)})[/green]'
        )
        return trainer

    except FileNotFoundError as e:
        console.print(f'\n[red][✗] {e}[/red]')
        sys.exit(1)
    except Exception as e:
        console.print(
            f'\n[red][✗] Model load failed: {e}[/red]'
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    """
    Start CICIDS2017 flow-level live IDS with Rich dashboard.

    1. Load CICIDS2017 adversarial model + scaler
    2. Start packet capture thread (Scapy)
    3. Start timeout monitor thread (background)
    4. Render Rich dashboard (main thread, 1s refresh)
    5. On Ctrl+C: print session summary
    """

    # ── Load CICIDS2017 model ─────────────────────────────────
    trainer = load_cicids_model()

    console.print(
        f'\n[green][✓] CICIDS2017 Flow IDS starting...[/green]'
    )
    console.print(
        f'[cyan]    Interface   : '
        f'{INTERFACE or "Auto-detect"}[/cyan]'
    )
    console.print(
        f'[cyan]    Flow timeout: {FLOW_TIMEOUT}s[/cyan]'
    )
    console.print(
        f'[cyan]    Log file    : {LOG_FILE}[/cyan]'
    )
    console.print(
        f'[yellow]    Note: alerts appear AFTER flow ends '
        f'({FLOW_TIMEOUT}s max delay)[/yellow]'
    )
    console.print(
        f'[yellow]    Press Ctrl+C to stop[/yellow]\n'
    )

    # Brief pause to read startup messages
    time.sleep(1)

    # ── Start packet capture in background daemon thread ──────
    # daemon=True → stops automatically when main exits
    # store=False → packets not kept in memory
    # filter='ip'  → IP packets only
    def capture_thread():
        sniff(
            iface  = INTERFACE,
            prn    = lambda pkt: packet_handler(pkt, trainer),
            store  = False,
            filter = 'ip'
        )

    pkt_thread = threading.Thread(
        target = capture_thread,
        daemon = True
    )
    pkt_thread.start()

    # ── Start timeout monitor in background daemon thread ─────
    # Classifies flows that end without FIN/RST
    # Essential for detecting DoS/scan flows
    tmr_thread = threading.Thread(
        target = timeout_monitor,
        args   = (trainer,),
        daemon = True
    )
    tmr_thread.start()

    # ── Rich Live Dashboard ───────────────────────────────────
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
            f'  Packets captured : '
            f'[white]{stats["packet_count"]:,}[/white]'
        )
        console.print(
            f'  Flows classified : '
            f'[white]{stats["flow_count"]:,}[/white]'
        )
        console.print(
            f'  Active flows     : '
            f'[yellow]{stats["active_flows"]:,}[/yellow]'
        )
        console.print(
            f'  Normal flows     : '
            f'[green]{stats["normal_count"]:,}[/green]'
        )
        console.print(
            f'  Attack flows     : '
            f'[red]{stats["attack_count"]:,}[/red]'
        )
        console.print(
            f'  Total alerts     : '
            f'[bold red]{stats["alert_count"]}[/bold red]'
        )
        console.print(
            f'  Log saved to     : '
            f'[cyan]{LOG_FILE}[/cyan]'
        )
        console.print(
            f'\n[green][✓] Goodbye![/green]\n'
        )


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    main()