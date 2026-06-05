"""
fix_model.py — AgroVision Pro
One-time utility: patch model.h5 to be compatible with older TensorFlow
that does not recognise 'groups': 1 in DepthwiseConv2D / Conv2D configs.

Run ONCE from the project root:
    python fix_model.py

Produces:
    model_fixed.h5   ← use this as your model.h5 going forward

Why this happens:
    TensorFlow ≥ 2.9 added a 'groups' parameter to DepthwiseConv2D.
    When a model is saved with TF 2.9+ and loaded with TF 2.6–2.8 (common
    on Streamlit Cloud / HuggingFace free tiers), Keras raises:
        "Unrecognized keyword arguments passed to DepthwiseConv2D: {'groups': 1}"
"""

import os
import re
import io
import sys
import json
import h5py
import numpy as np

INPUT_PATH  = "model.h5"
OUTPUT_PATH = "model_fixed.h5"


# ─── Approach A: h5py direct JSON patch ───────────────────────────────────────
def patch_via_h5py(src: str, dst: str) -> bool:
    """
    Open the HDF5 file, locate the model_config attribute, strip the
    'groups' key from every layer config, and write to dst.
    """
    try:
        import shutil
        shutil.copy2(src, dst)

        with h5py.File(dst, "a") as f:
            cfg_bytes = f.attrs.get("model_config", None)
            if cfg_bytes is None:
                print("  ⚠ No 'model_config' attr found — trying raw byte patch.")
                return False

            if isinstance(cfg_bytes, bytes):
                cfg_str = cfg_bytes.decode("utf-8")
            else:
                cfg_str = str(cfg_bytes)

            # Parse, patch recursively, re-serialise
            cfg = json.loads(cfg_str)
            _strip_groups(cfg)
            patched = json.dumps(cfg)

            f.attrs["model_config"] = patched.encode("utf-8")

        print(f"  ✅ h5py patch succeeded → {dst}")
        return True
    except Exception as exc:
        print(f"  ⚠ h5py patch failed: {exc}")
        return False


def _strip_groups(obj):
    """Recursively remove 'groups' from all layer configs."""
    if isinstance(obj, dict):
        obj.pop("groups", None)
        for v in obj.values():
            _strip_groups(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_groups(item)


# ─── Approach B: raw byte-level regex patch ───────────────────────────────────
def patch_via_bytes(src: str, dst: str) -> bool:
    """
    Read raw bytes, strip every occurrence of '"groups": <int>' via regex,
    write to dst.  Works even if h5py can't parse the config attribute.
    """
    try:
        with open(src, "rb") as f:
            raw = bytearray(f.read())

        original_len = len(raw)
        patched = re.sub(rb',?\s*"groups":\s*\d+', b"", raw)
        patched = re.sub(rb',?\s*"batch_shape":\s*null', b"", patched)

        with open(dst, "wb") as f:
            f.write(patched)

        removed = original_len - len(patched)
        print(f"  ✅ Byte-level patch succeeded ({removed} bytes removed) → {dst}")
        return True
    except Exception as exc:
        print(f"  ⚠ Byte-level patch failed: {exc}")
        return False


# ─── Approach C: re-save via TF custom_objects ───────────────────────────────
def patch_via_resave(src: str, dst: str) -> bool:
    """
    Load with a compatibility shim, then re-save cleanly.
    Requires TensorFlow to be importable.
    """
    try:
        import tensorflow as tf

        class CompatDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop("groups", None)
                kwargs.pop("batch_shape", None)
                super().__init__(*args, **kwargs)

        class CompatConv2D(tf.keras.layers.Conv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop("groups", None)
                kwargs.pop("batch_shape", None)
                super().__init__(*args, **kwargs)

        model = tf.keras.models.load_model(
            src,
            custom_objects={
                "DepthwiseConv2D": CompatDepthwiseConv2D,
                "Conv2D":          CompatConv2D,
            },
        )
        model.save(dst)
        print(f"  ✅ Re-save via TF succeeded → {dst}")
        return True
    except Exception as exc:
        print(f"  ⚠ Re-save via TF failed: {exc}")
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  AgroVision Pro — Model Compatibility Patcher")
    print("=" * 60)

    if not os.path.exists(INPUT_PATH):
        print(f"\n❌ '{INPUT_PATH}' not found. Place model.h5 in the same folder.")
        sys.exit(1)

    size_mb = os.path.getsize(INPUT_PATH) / 1_048_576
    print(f"\n  Input : {INPUT_PATH}  ({size_mb:.1f} MB)")
    print(f"  Output: {OUTPUT_PATH}\n")

    # Try approaches in order of preference
    print("→ Attempt 1: h5py JSON config patch …")
    if patch_via_h5py(INPUT_PATH, OUTPUT_PATH):
        _verify(OUTPUT_PATH)
        return

    print("\n→ Attempt 2: Raw byte-level regex patch …")
    if patch_via_bytes(INPUT_PATH, OUTPUT_PATH):
        _verify(OUTPUT_PATH)
        return

    print("\n→ Attempt 3: Load + re-save via TensorFlow …")
    if patch_via_resave(INPUT_PATH, OUTPUT_PATH):
        _verify(OUTPUT_PATH)
        return

    print("\n❌ All patch attempts failed.")
    print("   Please upgrade TensorFlow: pip install --upgrade tensorflow")
    sys.exit(1)


def _verify(path: str):
    """Quick verification that the patched file loads without error."""
    print(f"\n  Verifying {path} …")
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        out   = model.predict(dummy, verbose=0)
        print(f"  ✅ Verification passed — output shape: {out.shape}")
        print(f"\n  ► Replace your model.h5:\n"
              f"      cp {path} model.h5\n"
              f"  Then restart the Streamlit app.\n")
    except Exception as exc:
        print(f"  ⚠ Verification failed: {exc}")
        print("     The patched file may still work at runtime via the shim in app.py.")


if __name__ == "__main__":
    main()
