# =============================================================
# test_traffic.py
# =============================================================
# PURPOSE:
#   Generates synthetic normal and attack network traffic
#   locally inside WSL to test live_ids.py without needing
#   Kali or external network access.
#
# WHY THIS EXISTS:
#   WSL runs in NAT mode — external machines (Kali) cannot
#   reach WSL directly. This script simulates both normal
#   and attack traffic patterns from within WSL itself.
#
# HOW IT WORKS:
#   Uses Scapy to craft and send raw IP/TCP packets to
#   localhost (127.0.0.1). live_ids.py captures these
#   packets and classifies them in real time.
#
# TRAFFIC PATTERNS SIMULATED:
#   Normal  : Low packet rate, small src_bytes, standard ports
#             (HTTP:80, HTTPS:443, DNS:53)
#   Attack  : High packet rate, large src_bytes, SYN floods,
#             port scans, unusual flag combinations
#
# HOW TO USE:
#   Terminal 1: sudo ../venv/bin/python3 live_ids.py
#   Terminal 2: sudo ../venv/bin/python3 ../test_traffic.py
#
# NOTE: sudo required for raw packet crafting with Scapy
#
# CDAC ITISS — Adversarial Input Attack on ML-based IDS
# =============================================================

import time
import random
import sys
from scapy.all import (
    IP, TCP, UDP, send, conf
)

# Suppress Scapy output
conf.verb = 0


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Target IP — localhost (WSL internal)
TARGET_IP = '127.0.0.1'
SOURCE_IP = '127.0.0.1'

# Ports used in normal traffic simulation
NORMAL_PORTS = [80, 443, 53, 22, 8080, 3306]

# Ports targeted in attack simulation
ATTACK_PORTS = list(range(1, 1024))


# ─────────────────────────────────────────────────────────────
#  NORMAL TRAFFIC GENERATORS
# ─────────────────────────────────────────────────────────────

def send_normal_http():
    """
    Simulate normal HTTP traffic.
    Low packet rate, standard ports, small payload.
    """
    pkt = (
        IP(src=SOURCE_IP, dst=TARGET_IP) /
        TCP(
            sport=random.randint(1024, 65535),
            dport=80,
            flags='S'        # SYN — normal connection start
        )
    )
    send(pkt, verbose=False)
    print(f"  [NORMAL] HTTP  SYN  → {TARGET_IP}:80")


def send_normal_https():
    """Simulate normal HTTPS traffic."""
    pkt = (
        IP(src=SOURCE_IP, dst=TARGET_IP) /
        TCP(
            sport=random.randint(1024, 65535),
            dport=443,
            flags='S'
        )
    )
    send(pkt, verbose=False)
    print(f"  [NORMAL] HTTPS SYN  → {TARGET_IP}:443")


def send_normal_dns():
    """Simulate normal DNS query."""
    pkt = (
        IP(src=SOURCE_IP, dst=TARGET_IP) /
        UDP(
            sport=random.randint(1024, 65535),
            dport=53
        )
    )
    send(pkt, verbose=False)
    print(f"  [NORMAL] DNS   UDP  → {TARGET_IP}:53")


def send_normal_ssh():
    """Simulate normal SSH connection."""
    pkt = (
        IP(src=SOURCE_IP, dst=TARGET_IP) /
        TCP(
            sport=random.randint(1024, 65535),
            dport=22,
            flags='S'
        )
    )
    send(pkt, verbose=False)
    print(f"  [NORMAL] SSH   SYN  → {TARGET_IP}:22")


# ─────────────────────────────────────────────────────────────
#  ATTACK TRAFFIC GENERATORS
# ─────────────────────────────────────────────────────────────

def send_syn_flood(n_packets: int = 20):
    """
    Simulate SYN flood DoS attack.

    SYN flood = send many TCP SYN packets without completing
    the handshake. Exhausts server connection table.

    NSL-KDD signature:
      - High count (many connections)
      - High serror_rate (many SYN errors)
      - Same destination port
    """
    print(f"\n  [ATTACK] SYN Flood → {TARGET_IP}:80 "
          f"({n_packets} packets)")

    for i in range(n_packets):
        pkt = (
            IP(
                src=f"192.168.{random.randint(1,254)}."
                    f"{random.randint(1,254)}",  # spoofed src
                dst=TARGET_IP
            ) /
            TCP(
                sport=random.randint(1024, 65535),
                dport=80,
                flags='S',          # SYN only — no ACK
                seq=random.randint(0, 2**32 - 1)
            )
        )
        send(pkt, verbose=False)
        time.sleep(0.02)    # 50 packets/sec

    print(f"  [ATTACK] SYN Flood complete")


def send_port_scan(n_ports: int = 30):
    """
    Simulate TCP port scan (like nmap -sS).

    Port scan signature:
      - High diff_srv_rate (many different ports)
      - Many RST/REJ responses
      - Short duration per connection
    """
    print(f"\n  [ATTACK] Port Scan → {TARGET_IP} "
          f"({n_ports} ports)")

    ports = random.sample(ATTACK_PORTS, min(n_ports, len(ATTACK_PORTS)))

    for port in ports:
        pkt = (
            IP(src=SOURCE_IP, dst=TARGET_IP) /
            TCP(
                sport=random.randint(1024, 65535),
                dport=port,
                flags='S'           # SYN scan
            )
        )
        send(pkt, verbose=False)
        time.sleep(0.03)

    print(f"  [ATTACK] Port scan complete")


def send_udp_flood(n_packets: int = 20):
    """
    Simulate UDP flood attack.

    Sends many UDP packets to random ports.
    Signature: high count, high rerror_rate.
    """
    print(f"\n  [ATTACK] UDP Flood → {TARGET_IP} "
          f"({n_packets} packets)")

    for i in range(n_packets):
        pkt = (
            IP(src=SOURCE_IP, dst=TARGET_IP) /
            UDP(
                sport=random.randint(1024, 65535),
                dport=random.randint(1, 65535)
            )
        )
        send(pkt, verbose=False)
        time.sleep(0.02)

    print(f"  [ATTACK] UDP flood complete")


def send_rst_attack(n_packets: int = 15):
    """
    Simulate RST injection attack.

    Sends RST packets to disrupt existing connections.
    Signature: high rerror_rate, rst_count.
    """
    print(f"\n  [ATTACK] RST Injection → {TARGET_IP} "
          f"({n_packets} packets)")

    for i in range(n_packets):
        pkt = (
            IP(src=SOURCE_IP, dst=TARGET_IP) /
            TCP(
                sport=random.randint(1024, 65535),
                dport=random.choice([80, 443, 22]),
                flags='R'           # RST flag
            )
        )
        send(pkt, verbose=False)
        time.sleep(0.03)

    print(f"  [ATTACK] RST injection complete")


def send_large_payload_attack(n_packets: int = 10):
    """
    Simulate large payload attack (data exfiltration / DoS).

    Sends packets with large payloads.
    NSL-KDD signature: very high src_bytes.
    """
    print(f"\n  [ATTACK] Large Payload → {TARGET_IP} "
          f"({n_packets} packets)")

    # 1400 bytes payload (near MTU limit)
    payload = b'X' * 1400

    for i in range(n_packets):
        pkt = (
            IP(src=SOURCE_IP, dst=TARGET_IP) /
            TCP(
                sport=random.randint(1024, 65535),
                dport=80,
                flags='PA'          # PSH + ACK (data transfer)
            ) /
            payload
        )
        send(pkt, verbose=False)
        time.sleep(0.05)

    print(f"  [ATTACK] Large payload complete")


# ─────────────────────────────────────────────────────────────
#  TRAFFIC SCENARIOS
# ─────────────────────────────────────────────────────────────

def run_normal_scenario(n_cycles: int = 5):
    """
    Run normal traffic pattern for n_cycles.
    Mix of HTTP, HTTPS, DNS, SSH traffic.
    """
    print(f"\n{'='*55}")
    print(f"  Running NORMAL traffic scenario ({n_cycles} cycles)")
    print(f"{'='*55}")

    for cycle in range(n_cycles):
        print(f"\n  Cycle {cycle + 1}/{n_cycles}")

        send_normal_http()
        time.sleep(0.3)

        send_normal_https()
        time.sleep(0.3)

        send_normal_dns()
        time.sleep(0.2)

        send_normal_ssh()
        time.sleep(0.5)

    print(f"\n  Normal scenario complete")


def run_attack_scenario():
    """
    Run all attack patterns in sequence.
    Simulates a full attack campaign.
    """
    print(f"\n{'='*55}")
    print(f"  Running ATTACK traffic scenario")
    print(f"{'='*55}")

    # Attack 1: SYN Flood (DoS)
    print(f"\n  [1/4] SYN Flood (DoS Attack)")
    send_syn_flood(n_packets=25)
    time.sleep(1)

    # Attack 2: Port Scan (Reconnaissance)
    print(f"\n  [2/4] Port Scan (Reconnaissance)")
    send_port_scan(n_ports=40)
    time.sleep(1)

    # Attack 3: UDP Flood
    print(f"\n  [3/4] UDP Flood")
    send_udp_flood(n_packets=25)
    time.sleep(1)

    # Attack 4: RST Injection
    print(f"\n  [4/4] RST Injection")
    send_rst_attack(n_packets=20)
    time.sleep(1)

    print(f"\n  Attack scenario complete")


def run_mixed_scenario():
    """
    Run mixed normal + attack traffic.
    Most realistic test — mimics real network.
    """
    print(f"\n{'='*55}")
    print(f"  Running MIXED traffic scenario")
    print(f"{'='*55}")

    # Start with some normal traffic
    print(f"\n  Phase 1: Normal baseline traffic")
    run_normal_scenario(n_cycles=3)
    time.sleep(2)

    # Then attack
    print(f"\n  Phase 2: Attack traffic begins")
    send_syn_flood(n_packets=20)
    time.sleep(0.5)

    # Normal traffic continues alongside attack
    print(f"\n  Phase 3: Mixed traffic")
    for _ in range(3):
        send_normal_http()
        time.sleep(0.2)
        send_port_scan(n_ports=15)
        time.sleep(0.5)
        send_normal_https()
        time.sleep(0.2)
        send_udp_flood(n_packets=10)
        time.sleep(0.5)

    print(f"\n  Mixed scenario complete")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    """
    Interactive traffic generator menu.
    Choose which traffic scenario to run.
    """
    print(f"""
{'='*55}
  Traffic Generator for Live IDS Testing
  CDAC ITISS — Adversarial Input Attack on ML-based IDS
{'='*55}

  Target : {TARGET_IP}

  Make sure live_ids.py is running in another terminal:
    sudo ../venv/bin/python3 src/live_ids.py

  Choose traffic scenario:
    1 → Normal traffic only
    2 → Attack traffic only  (triggers alerts)
    3 → Mixed traffic        (most realistic)
    4 → Continuous loop      (keeps sending until Ctrl+C)
""")

    choice = input("  Enter choice (1/2/3/4): ").strip()

    if choice == '1':
        run_normal_scenario(n_cycles=5)

    elif choice == '2':
        run_attack_scenario()

    elif choice == '3':
        run_mixed_scenario()

    elif choice == '4':
        print(f"\n  Starting continuous loop (Ctrl+C to stop)...")
        cycle = 0
        try:
            while True:
                cycle += 1
                print(f"\n{'─'*40}")
                print(f"  Cycle {cycle}")
                print(f"{'─'*40}")

                # Alternate between normal and attack
                if cycle % 3 == 0:
                    # Every 3rd cycle = attack
                    send_syn_flood(n_packets=15)
                    send_port_scan(n_ports=20)
                else:
                    # Other cycles = normal
                    run_normal_scenario(n_cycles=2)

                time.sleep(2)

        except KeyboardInterrupt:
            print(f"\n\n  Stopped after {cycle} cycles.")

    else:
        print(f"  Invalid choice. Run again and enter 1, 2, 3 or 4.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Traffic generation complete")
    print(f"  Check live_ids.py dashboard for alerts")
    print(f"{'='*55}\n")


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    main()