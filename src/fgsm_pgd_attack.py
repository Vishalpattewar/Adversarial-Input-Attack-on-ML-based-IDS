# =============================================================
# src/fgsm_pgd_attack.py
# =============================================================
# PURPOSE:
#   Implements FGSM and PGD adversarial attacks against the
#   trained IDSNet model on CICIDS2017 dataset.
#   Evaluates how well an attacker can evade the IDS by
#   perturbing only features the attacker can realistically
#   control (mutable features).
#
# WHAT THIS FILE DOES:
#   1. Imports NUMERIC_FEATURES + MUTABLE_FEATURE_NAMES
#      from ids_model.py (single source of truth)
#   2. Computes MUTABLE_INDICES automatically
#   3. FGSMAttacker class:
#        attack_fgsm()      — single-step gradient attack
#        attack_pgd()       — multi-step iterative attack
#        evaluate_evasion() — measure evasion rate at epsilon
#   4. epsilon_sweep()     — test across all epsilon values
#                            returns evasion rates for plotting
#
# ATTACK BACKGROUND:
#   FGSM (Fast Gradient Sign Method) — Goodfellow et al. 2014
#     Single step: x_adv = x + ε · sign(∇_x Loss(model(x), y))
#     Fast but weak — single step may not find best perturbation
#
#   PGD (Projected Gradient Descent) — Madry et al. 2017
#     Multi-step: iterate FGSM with smaller steps
#     Stronger attack — finds better adversarial examples
#     Standard for adversarial robustness evaluation
#
# MUTABLE vs IMMUTABLE FEATURES:
#   Mutable   = attacker (sender) controls these 27 features
#   Immutable = server response / network-observed 25 features
#
#   WHY THIS MATTERS ACADEMICALLY:
#     Perturbing server-response features is physically impossible
#     (attacker cannot change server's TCP window size)
#     Restricting to mutable features = realistic attack model
#     Makes adversarial analysis valid and defensible
#
# PHYSICAL CONSTRAINTS:
#   All perturbed features clipped to >= 0
#   WHY: negative packet counts, sizes, rates are impossible
#   Ensures adversarial examples are physically plausible
#   non_neg_idx = ALL 52 features (all must be non-negative)
#
# BUGS FIXED VS ORIGINAL NSL-KDD VERSION:
#   FIX 1: squeeze(-1) everywhere instead of squeeze()
#           bare squeeze() collapses (1,1)→scalar when batch_size=1
#           breaks attack on single samples
#
#   FIX 2: non_neg_idx = list(range(len(NUMERIC_FEATURES)))
#           original excluded rate features from non-negative clip
#           allowed physically impossible negative packet counts
#           ALL 52 features must be >= 0 (network statistics)
#
#   FIX 3: MUTABLE_INDICES auto-computed from imported lists
#           no hardcoding — stays correct if features change
#
#   FIX 4: __file__ anchored BASE_DIR
#           relative paths break if called from different CWD
#
# HOW TO USE:
#   from fgsm_pgd_attack import FGSMAttacker, epsilon_sweep
#   attacker = FGSMAttacker(model, scaler)
#   results  = epsilon_sweep(attacker, X_attack, EPSILONS)
#
# CDAC ITISS — Adversarial Input Attack on ML-based IDS
# =============================================================


import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

# ── Allow import from same src/ directory ─────────────────────
# Needed when run_experiment.py imports this file from src/
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from ids_model import (
    IDSNet,
    NUMERIC_FEATURES,
    MUTABLE_FEATURE_NAMES
)


# ─────────────────────────────────────────────────────────────
#  PATH CONFIGURATION
#
#  FIX 4: Anchor all paths to script location not CWD
#  WHY realpath: resolves symlinks → works in Docker/NFS/WSL
#  WHY try/except: __file__ undefined in Jupyter → fallback
# ─────────────────────────────────────────────────────────────
try:
    _BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    _BASE_DIR = os.getcwd()


# ─────────────────────────────────────────────────────────────
#  MUTABLE INDICES
#
#  Auto-computed from NUMERIC_FEATURES and MUTABLE_FEATURE_NAMES
#  Both imported from ids_model.py — single source of truth
#
#  FIX 3: Never hardcode indices
#    If NUMERIC_FEATURES order ever changes →
#    MUTABLE_INDICES updates automatically
#
#  MUTABLE_INDICES: list of integer positions in feature vector
#    that the attacker can perturb during the attack
#
#  IMMUTABLE positions (not in MUTABLE_INDICES):
#    Bwd Packet Length features  → server controls response
#    Bwd IAT features            → server timing
#    Bwd Header Length           → server header
#    Bwd Packets/s               → server packet rate
#    Packet Length stats         → combined fwd+bwd (server side)
#    ACK Flag Count              → largely server response
#    Average Packet Size         → combined statistic
#    Init_Win_bytes_backward     → server TCP window
# ─────────────────────────────────────────────────────────────
MUTABLE_INDICES = [
    NUMERIC_FEATURES.index(f)
    for f in MUTABLE_FEATURE_NAMES
]
# Total: 27 mutable indices out of 52


# ─────────────────────────────────────────────────────────────
#  PHYSICAL CONSTRAINT INDICES
#
#  FIX 2: ALL features must be non-negative
#  Network statistics (packet counts, sizes, rates, times)
#  cannot be negative in real network traffic
#
#  Original NSL-KDD version only clipped some features →
#  allowed negative packet lengths and counts → bug
#
#  non_neg_idx = ALL feature indices = clip everything to >= 0
# ─────────────────────────────────────────────────────────────
NON_NEG_IDX = list(range(len(NUMERIC_FEATURES)))
# All 52 features clipped to >= 0 after perturbation


# ─────────────────────────────────────────────────────────────
#  FGSMAttacker — FGSM and PGD Attack Implementation
# ─────────────────────────────────────────────────────────────
class FGSMAttacker:
    """
    Adversarial attack engine for IDSNet on CICIDS2017.

    Implements FGSM and PGD attacks restricted to
    physically realistic mutable features only.

    FGSM: Fast single-step gradient attack
    PGD : Stronger multi-step iterative attack

    Args:
        model  : Trained IDSNet in eval() mode
        scaler : Fitted StandardScaler from IDSTrainer
                 Used to scale inputs before model forward pass
    """

    def __init__(self, model: IDSNet, scaler):
        """
        Initialise attacker with trained model and scaler.

        Args:
            model : IDSNet instance, must be in eval() mode
            scaler: StandardScaler fitted on X_train
        """
        self.model  = model
        self.scaler = scaler

        # Loss function — same as training
        # WHY BCELoss: model uses Sigmoid output → BCELoss correct
        self.criterion = nn.BCELoss()

        # Ensure model is in eval mode
        # eval() disables Dropout → deterministic output
        self.model.eval()

    # ──────────────────────────────────────────────────────────
    def _scale(self, X: np.ndarray) -> torch.Tensor:
        """
        Scale raw features using fitted StandardScaler.

        Args:
            X: Raw feature array shape (N, 52)

        Returns:
            Scaled tensor shape (N, 52) dtype float32
        """
        X_scaled = self.scaler.transform(
            X.astype(np.float32)
        )
        return torch.tensor(X_scaled, dtype=torch.float32)

    # ──────────────────────────────────────────────────────────
    def _unscale(self, X_tensor: torch.Tensor) -> np.ndarray:
        """
        Inverse scale tensor back to raw feature space.

        WHY: Perturbations applied in scaled space
             Physical constraints (>= 0) applied in raw space
             Must unscale → clip → rescale for physical validity

        Args:
            X_tensor: Scaled tensor shape (N, 52)

        Returns:
            Raw feature array shape (N, 52) dtype float32
        """
        return self.scaler.inverse_transform(
            X_tensor.detach().numpy().astype(np.float32)
        )

    # ──────────────────────────────────────────────────────────
    def attack_fgsm(self,
                    X_raw    : np.ndarray,
                    y        : np.ndarray,
                    epsilon  : float) -> np.ndarray:
        """
        Fast Gradient Sign Method (FGSM) attack.

        Single-step attack: moves inputs in the direction
        that maximises classification loss.

        Formula:
          x_adv = x + ε · sign(∇_x Loss(f(x), y))

        Only perturbs MUTABLE_INDICES (27 features).
        Clips all features to >= 0 after perturbation.

        FIX 1: squeeze(-1) not squeeze()
        FIX 2: clip ALL features (NON_NEG_IDX) to >= 0

        Args:
            X_raw  : Raw attack samples shape (N, 52)
                     Should be attack-class samples (label=1)
            y      : True labels shape (N,) values = 1.0
            epsilon: Perturbation budget
                     Larger = stronger attack but less realistic
                     Range tested: [0.05, 0.10, 0.20, 0.50,
                                    0.75, 1.00, 1.50, 2.00]

        Returns:
            X_adv_raw: Adversarial samples in raw feature space
                       shape (N, 52) dtype float32
                       Immutable features unchanged
                       Mutable features perturbed by ε
        """
        # Scale inputs to model space
        X_tensor = self._scale(X_raw)
        X_tensor.requires_grad_(True)

        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Forward pass
        # FIX 1: squeeze(-1) not squeeze()
        out  = self.model(X_tensor).squeeze(-1)
        loss = self.criterion(out, y_tensor)

        # Backward pass — compute gradient w.r.t. INPUT
        # We want ∇_x Loss not ∇_θ Loss
        loss.backward()

        # Compute FGSM perturbation in scaled space
        # sign() gives ±1 per feature
        perturbation = epsilon * X_tensor.grad.sign()

        # Apply perturbation ONLY to mutable feature indices
        # Immutable features stay exactly as original
        X_adv = X_tensor.clone().detach()
        X_adv[:, MUTABLE_INDICES] = (
            X_tensor[:, MUTABLE_INDICES] +
            perturbation[:, MUTABLE_INDICES]
        )

        # Convert back to raw feature space
        X_adv_raw = self._unscale(X_adv)

        # FIX 2: Clip ALL features to >= 0
        # Physical constraint: no negative network statistics
        # Ensures adversarial examples are physically plausible
        X_adv_raw[:, NON_NEG_IDX] = np.clip(
            X_adv_raw[:, NON_NEG_IDX], a_min=0.0, a_max=None
        )

        return X_adv_raw.astype(np.float32)

    # ──────────────────────────────────────────────────────────
    def attack_pgd(self,
                   X_raw    : np.ndarray,
                   y        : np.ndarray,
                   epsilon  : float,
                   n_steps  : int   = 10,
                   step_size: float = None) -> np.ndarray:
        """
        Projected Gradient Descent (PGD) attack.

        Multi-step iterative attack — stronger than FGSM.
        Applies FGSM repeatedly with smaller steps.
        Projects back to epsilon ball after each step.

        Formula (per step):
          x_t+1 = Proj(x_t + α · sign(∇_x Loss(f(x_t), y)))
          where α = epsilon / n_steps (step size)
          Proj = project back to ε-ball around original x

        WHY PGD STRONGER THAN FGSM:
          FGSM takes one large step → may miss optimal direction
          PGD iterates small steps → finds better perturbation
          Expected to show more evasion on CICIDS2017 than
          NSL-KDD (more complex non-linear decision boundary)

        FIX 1: squeeze(-1) not squeeze()
        FIX 2: clip ALL features to >= 0 after each step

        Args:
            X_raw    : Raw attack samples shape (N, 52)
            y        : True labels shape (N,)
            epsilon  : Total perturbation budget
            n_steps  : Number of PGD steps (default 10)
            step_size: Step size per iteration
                       Default: epsilon / n_steps
                       Smaller steps = finer search

        Returns:
            X_adv_raw: Adversarial samples shape (N, 52)
                       Stronger than FGSM at same epsilon
        """
        # Default step size: divide budget equally across steps
        if step_size is None:
            step_size = epsilon / n_steps

        # Scale inputs to model space
        X_tensor = self._scale(X_raw)

        # Store original scaled inputs for projection
        # Projection keeps perturbation within epsilon ball
        X_orig = X_tensor.clone().detach()

        # Initialise adversarial example as clean input
        X_adv  = X_tensor.clone().detach()

        # ── PGD iteration loop ─────────────────────────────────
        for step in range(n_steps):

            # Enable gradient computation for this step
            X_adv.requires_grad_(True)

            y_tensor = torch.tensor(y, dtype=torch.float32)

            # Forward pass
            # FIX 1: squeeze(-1) not squeeze()
            out  = self.model(X_adv).squeeze(-1)
            loss = self.criterion(out, y_tensor)

            # Backward pass
            loss.backward()

            # Gradient sign step (small step_size not full epsilon)
            with torch.no_grad():
                grad_sign = X_adv.grad.sign()

                # Apply step ONLY to mutable features
                X_adv_step = X_adv.clone()
                X_adv_step[:, MUTABLE_INDICES] = (
                    X_adv[:, MUTABLE_INDICES] +
                    step_size * grad_sign[:, MUTABLE_INDICES]
                )

                # Project back to epsilon ball around original input
                # Ensures total perturbation <= epsilon
                # Only project mutable features
                delta = X_adv_step - X_orig
                delta[:, MUTABLE_INDICES] = torch.clamp(
                    delta[:, MUTABLE_INDICES],
                    min = -epsilon,
                    max =  epsilon
                )

                # Immutable features: zero perturbation always
                immutable_idx = [
                    i for i in range(len(NUMERIC_FEATURES))
                    if i not in MUTABLE_INDICES
                ]
                delta[:, immutable_idx] = 0.0

                # Update adversarial example
                X_adv = (X_orig + delta).detach()

        # Convert back to raw feature space
        X_adv_raw = self._unscale(X_adv)

        # FIX 2: Clip ALL features to >= 0 after all steps
        # Physical constraint enforced after each iteration
        X_adv_raw[:, NON_NEG_IDX] = np.clip(
            X_adv_raw[:, NON_NEG_IDX], a_min=0.0, a_max=None
        )

        return X_adv_raw.astype(np.float32)

    # ──────────────────────────────────────────────────────────
    def evaluate_evasion(self,
                         X_raw  : np.ndarray,
                         y      : np.ndarray,
                         X_adv  : np.ndarray) -> dict:
        """
        Measure evasion rate of adversarial examples.

        Evasion rate = fraction of attack samples that the
        model classifies as NORMAL after perturbation.

        High evasion rate = attack is effective
        Low evasion rate  = model is robust at this epsilon

        Args:
            X_raw: Original clean attack samples (N, 52)
            y    : True labels (all should be 1 = Attack)
            X_adv: Adversarially perturbed samples (N, 52)

        Returns:
            dict with keys:
              clean_acc    : model accuracy on clean samples
              adv_acc      : model accuracy on adversarial
              evasion_rate : fraction successfully evaded
              n_samples    : number of samples evaluated
        """
        # Evaluate on clean samples
        X_clean_t  = self._scale(X_raw)
        y_tensor   = torch.tensor(y, dtype=torch.float32)

        with torch.no_grad():
            # FIX 1: squeeze(-1) not squeeze()
            clean_out   = self.model(X_clean_t).squeeze(-1)
            clean_preds = (clean_out >= 0.5).float()
            clean_acc   = (
                (clean_preds == y_tensor).float().mean().item()
            )

        # Evaluate on adversarial samples
        X_adv_t = self._scale(X_adv)

        with torch.no_grad():
            # FIX 1: squeeze(-1) not squeeze()
            adv_out   = self.model(X_adv_t).squeeze(-1)
            adv_preds = (adv_out >= 0.5).float()
            adv_acc   = (
                (adv_preds == y_tensor).float().mean().item()
            )

        # Evasion rate = attacks classified as Normal (0)
        # = 1 - accuracy on adversarial samples
        evasion_rate = 1.0 - adv_acc

        return {
            'clean_acc'   : round(clean_acc    * 100, 2),
            'adv_acc'     : round(adv_acc       * 100, 2),
            'evasion_rate': round(evasion_rate  * 100, 2),
            'n_samples'   : len(X_raw)
        }


# ─────────────────────────────────────────────────────────────
#  EPSILON SWEEP
#
#  Runs attacks at all epsilon values and collects results.
#  Used by run_experiment.py for both models and both attacks.
#
#  WHY SWEEP:
#    Single epsilon tells us nothing about robustness curve
#    Sweeping shows HOW the model degrades under increasing attack
#    Used to plot evasion_curves.png in visualise_results.py
# ─────────────────────────────────────────────────────────────
def epsilon_sweep(attacker  : FGSMAttacker,
                  X_attack  : np.ndarray,
                  y_attack  : np.ndarray,
                  epsilons  : list,
                  attack_type: str = 'fgsm',
                  n_pgd_steps: int = 10) -> dict:
    """
    Run FGSM or PGD attack across all epsilon values.

    For each epsilon:
      1. Generate adversarial examples
      2. Evaluate evasion rate
      3. Record results

    Args:
        attacker   : FGSMAttacker instance
        X_attack   : Raw attack-class samples shape (N, 52)
                     Should only contain label=1 samples
        y_attack   : True labels shape (N,) all = 1.0
        epsilons   : List of epsilon values to test
                     [0.05, 0.10, 0.20, 0.50, 0.75,
                      1.00, 1.50, 2.00]
        attack_type: 'fgsm' or 'pgd'
        n_pgd_steps: Number of PGD iterations (default 10)
                     Only used when attack_type='pgd'

    Returns:
        results dict with structure:
          {
            'attack_type': 'fgsm' or 'pgd',
            'n_samples'  : int,
            'epsilons'   : [list of epsilon values],
            'evasion_rates': [list of evasion % per epsilon],
            'clean_acc'  : float (baseline on clean samples),
            'per_epsilon': {
                '0.05': {'evasion_rate': x, 'adv_acc': y, ...},
                '0.10': {...},
                ...
            }
          }
    """
    print(f'\n{"─" * 55}')
    print(f'  {attack_type.upper()} Epsilon Sweep')
    print(f'  Samples : {len(X_attack):,}')
    print(f'  Epsilons: {epsilons}')
    if attack_type == 'pgd':
        print(f'  PGD steps: {n_pgd_steps}')
    print(f'{"─" * 55}')

    results = {
        'attack_type'  : attack_type,
        'n_samples'    : len(X_attack),
        'epsilons'     : epsilons,
        'evasion_rates': [],
        'clean_acc'    : 0.0,
        'per_epsilon'  : {}
    }

    for eps in epsilons:

        # ── Generate adversarial examples ─────────────────────
        if attack_type == 'fgsm':
            X_adv = attacker.attack_fgsm(
                X_attack, y_attack, epsilon=eps
            )
        elif attack_type == 'pgd':
            X_adv = attacker.attack_pgd(
                X_attack, y_attack,
                epsilon  = eps,
                n_steps  = n_pgd_steps
            )
        else:
            raise ValueError(
                f"attack_type must be 'fgsm' or 'pgd', "
                f"got '{attack_type}'"
            )

        # ── Evaluate evasion rate ─────────────────────────────
        eval_result = attacker.evaluate_evasion(
            X_attack, y_attack, X_adv
        )

        # Store baseline clean accuracy once
        if results['clean_acc'] == 0.0:
            results['clean_acc'] = eval_result['clean_acc']

        # Record evasion rate for this epsilon
        results['evasion_rates'].append(
            eval_result['evasion_rate']
        )
        results['per_epsilon'][str(eps)] = eval_result

        # Progress output
        print(f'  ε={eps:<5}  '
              f'Clean: {eval_result["clean_acc"]:5.1f}%  |  '
              f'Adv: {eval_result["adv_acc"]:5.1f}%  |  '
              f'Evasion: {eval_result["evasion_rate"]:5.1f}%')

    print(f'{"─" * 55}')
    print(f'  Max evasion: '
          f'{max(results["evasion_rates"]):.1f}% '
          f'at ε={epsilons[results["evasion_rates"].index(max(results["evasion_rates"]))]}')

    return results


# ─────────────────────────────────────────────────────────────
#  SAVE RESULTS
#
#  Saves epsilon sweep results to JSON file.
#  Called by run_experiment.py after each sweep.
# ─────────────────────────────────────────────────────────────
def save_results(results   : dict,
                 output_path: str):
    """
    Save epsilon sweep results to JSON file.

    Args:
        results    : dict returned by epsilon_sweep()
        output_path: Full path to output JSON file
                     e.g. results/cicids/standard/
                          attack_results_fgsm.json
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'  Results saved → {output_path}')