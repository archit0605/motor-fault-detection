"""
Synthetic vibration signal generator for four motor conditions.

Every recording is built the same way a real accelerometer trace would be
explained in an SnS course: a sum of harmonically related sinusoids (the
Fourier series of a periodic rotating-machine signal) plus additive white
Gaussian noise, with one class carrying an extra impulse-train term that
models a localized bearing defect as the impulse response of a resonant
structure being repeatedly struck.

Shaft speed, phases, noise level and harmonic amplitudes are all randomized
per recording so the four classes overlap somewhat in feature space, the
way real machines do -- a classifier that hits 100% on this data is
overfitting to an artifact, not learning the physics.
"""

import numpy as np

SEED = 42
FS = 12000          # sampling rate, Hz
DURATION = 2.0       # seconds per recording
F_SHAFT_NOMINAL = 30.0   # Hz, i.e. 1800 RPM
N_PER_CLASS = 40
BPFO_RATIO = 3.5     # ball-pass frequency, outer race, as a multiple of shaft speed
RESONANCE_FREQ = 3000.0  # Hz, natural frequency of the bearing/housing structure excited by each impact
RESONANCE_ZETA = 0.05     # damping ratio of that resonance (underdamped -> "ringing")

CLASSES = ["healthy", "imbalance", "misalignment", "bearing_fault"]


def _time_vector():
    return np.arange(0, DURATION, 1.0 / FS)


def _harmonic_sum(t, f_shaft, amps, phases):
    """1x/2x/3x shaft-harmonic content -- the truncated Fourier series of a
    periodic rotor signature. amps = [a1, a2, a3]."""
    x = np.zeros_like(t)
    for k, (a, phi) in enumerate(zip(amps, phases), start=1):
        x += a * np.sin(2 * np.pi * k * f_shaft * t + phi)
    return x


def _resonance_kernel(rng):
    """Impulse response of an underdamped 2nd-order LTI system:
    h(t) = exp(-zeta*wn*t) * sin(wd*t). This is what a bearing impact
    'rings out' as -- the housing/bearing behaves like a mass-spring-damper
    struck by a hammer. Kernel length is truncated once the envelope decays
    to ~1% so the convolution below stays cheap.
    """
    wn = 2 * np.pi * RESONANCE_FREQ
    zeta = RESONANCE_ZETA
    wd = wn * np.sqrt(1 - zeta ** 2)
    t_decay = -np.log(0.01) / (zeta * wn)
    n_samples = int(t_decay * FS)
    tk = np.arange(n_samples) / FS
    kernel = np.exp(-zeta * wn * tk) * np.sin(wd * tk)
    return kernel


def _bearing_impulse_train(t, f_shaft, amp_res, rng):
    """Periodic impulse train at BPFO, each impulse exciting the resonance
    kernel above. Real bearing faults have small random jitter in impact
    timing (the balls don't strike at a perfectly fixed rate) and small
    impact-to-impact amplitude variation (load/contact-angle variation), so
    both are randomized here. Built as delta-train * kernel (convolution)
    rather than looping in Python -- much faster and it's literally what
    convolution means physically: sum of shifted, scaled impulse responses.
    """
    bpfo = BPFO_RATIO * f_shaft
    period = 1.0 / bpfo
    n_impulses = int(DURATION / period)

    impulse_times = (np.arange(n_impulses) + 1) * period
    jitter = rng.uniform(-0.02, 0.02, n_impulses) * period
    impulse_times += jitter
    impulse_amplitudes = amp_res * rng.uniform(0.8, 1.2, n_impulses)

    delta_train = np.zeros_like(t)
    idx = np.round(impulse_times * FS).astype(int)
    valid = (idx >= 0) & (idx < len(t))
    np.add.at(delta_train, idx[valid], impulse_amplitudes[valid])

    kernel = _resonance_kernel(rng)
    ringing = np.convolve(delta_train, kernel, mode="full")[: len(t)]
    return ringing, bpfo


def generate_one(fault_type, seed):
    """Generate a single recording for the given class and RNG seed.
    Returns (signal, fault_type, metadata)."""
    rng = np.random.default_rng(seed)
    t = _time_vector()

    f_shaft = F_SHAFT_NOMINAL * (1 + rng.uniform(-0.05, 0.05))
    phases = rng.uniform(0, 2 * np.pi, 3)
    meta = {"f_shaft": f_shaft, "fs": FS, "seed": seed, "label": fault_type}

    if fault_type == "healthy":
        amps = [rng.uniform(0.06, 0.18), rng.uniform(0.01, 0.05), rng.uniform(0.005, 0.02)]
        noise_std = rng.uniform(0.10, 0.28)
        x = _harmonic_sum(t, f_shaft, amps, phases)

    elif fault_type == "imbalance":
        # unbalance force scales with rotational speed squared (F = m*e*omega^2),
        # so the 1x amplitude is driven by f_shaft**2, not chosen freely. The
        # random multiplier is wide enough that a weak imbalance can look a
        # lot like a healthy machine, and a strong one stands well apart --
        # real unbalance severity varies a lot machine to machine
        speed_ratio_sq = (f_shaft / F_SHAFT_NOMINAL) ** 2
        a1 = 0.45 * speed_ratio_sq * rng.uniform(0.4, 1.2)
        amps = [a1, rng.uniform(0.02, 0.08), rng.uniform(0.01, 0.04)]
        noise_std = rng.uniform(0.10, 0.28)
        x = _harmonic_sum(t, f_shaft, amps, phases)

    elif fault_type == "misalignment":
        # textbook misalignment signature: strong 2x, appreciable 3x, 1x present but not dominant
        amps = [rng.uniform(0.08, 0.20), rng.uniform(0.15, 0.40), rng.uniform(0.06, 0.20)]
        noise_std = rng.uniform(0.10, 0.28)
        x = _harmonic_sum(t, f_shaft, amps, phases)

    elif fault_type == "bearing_fault":
        # low-frequency rotor content stays small on purpose -- the fault
        # energy lives up at the resonance and is invisible in a raw low-band look
        amps = [rng.uniform(0.03, 0.09), rng.uniform(0.01, 0.03), rng.uniform(0.005, 0.02)]
        noise_std = rng.uniform(0.10, 0.28)
        x = _harmonic_sum(t, f_shaft, amps, phases)
        # amplitude varies from a barely-there early-stage fault to a severe
        # one, so envelope-domain separability from "healthy" is not trivial
        amp_res = rng.uniform(0.15, 0.65)
        ringing, bpfo = _bearing_impulse_train(t, f_shaft, amp_res, rng)
        x = x + ringing
        meta["bpfo"] = bpfo

    else:
        raise ValueError(f"unknown fault_type: {fault_type}")

    x = x + rng.normal(0, noise_std, len(t))
    meta["noise_std"] = noise_std
    return x, fault_type, meta


def generate_dataset(n_per_class=N_PER_CLASS, seed=SEED):
    """Build the full labelled dataset: n_per_class recordings for each of
    the 4 classes. Returns a list of (signal, label, metadata) tuples."""
    ss = np.random.SeedSequence(seed)
    class_seeds = ss.spawn(len(CLASSES))

    recordings = []
    for label, class_seed in zip(CLASSES, class_seeds):
        rec_seeds = class_seed.spawn(n_per_class)
        for i, rec_seed in enumerate(rec_seeds):
            child_seed = np.random.default_rng(rec_seed).integers(0, 2**32 - 1)
            x, lbl, meta = generate_one(label, int(child_seed))
            meta["recording_id"] = f"{label}_{i:03d}"
            recordings.append((x, lbl, meta))
    return recordings


if __name__ == "__main__":
    data = generate_dataset()
    print(f"Generated {len(data)} recordings across {len(CLASSES)} classes.")
    x, label, meta = data[0]
    print(f"Example: label={label}, len={len(x)} samples, f_shaft={meta['f_shaft']:.2f} Hz")
