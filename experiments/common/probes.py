"""Runtime probes shared by the runners: peak GPU memory (in MB) and atomic checkpoint saves."""
import os

import equinox as eqx


def peak_gpu_mb():
    """Peak device bytes-in-use in MB (honest single-device footprint, not the preallocated pool),
    or −1.0 if the stat is unavailable (CPU / older JAX)."""
    try:
        import jax
        return jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 1e6
    except Exception:
        return -1.0


def atomic_save(path, tree):
    """Serialize an Equinox pytree to ``path`` atomically (write ``path.tmp`` then ``os.replace``),
    so a preempted checkpoint never leaves a half-written file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    eqx.tree_serialise_leaves(path + ".tmp", tree)
    os.replace(path + ".tmp", path)
