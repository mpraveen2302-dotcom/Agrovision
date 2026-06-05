"""
train_efficientnet.py — AgroVision Pro
EfficientNetB0 Training Pipeline with:
  Transfer Learning → Fine-Tuning
  Mixed Precision
  Data Augmentation
  EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# ─── Mixed precision ──────────────────────────────────────────────────────────
tf.keras.mixed_precision.set_global_policy("mixed_float16")


# ─── CLI Arguments ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train AgroVision EfficientNetB0")
    p.add_argument("--data_dir",    default="dataset",       help="Root dataset folder (train/val subdirs)")
    p.add_argument("--model_out",   default="model.h5",      help="Output model path")
    p.add_argument("--classes_out", default="class_names.json")
    p.add_argument("--epochs_tl",   type=int, default=20,    help="Transfer-learning epochs (frozen backbone)")
    p.add_argument("--epochs_ft",   type=int, default=30,    help="Fine-tuning epochs (unfrozen backbone)")
    p.add_argument("--batch",       type=int, default=32)
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--lr_tl",       type=float, default=1e-3)
    p.add_argument("--lr_ft",       type=float, default=1e-5)
    p.add_argument("--unfreeze_at", type=int, default=100,   help="Unfreeze from this layer index during fine-tuning")
    return p.parse_args()


# ─── Augmentation Pipeline ────────────────────────────────────────────────────
def build_augmentation_model(img_size: int) -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")


# ─── Data Loaders ─────────────────────────────────────────────────────────────
def load_datasets(data_dir: str, img_size: int, batch: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=(img_size, img_size),
        batch_size=batch,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=(img_size, img_size),
        batch_size=batch,
        label_mode="categorical",
        shuffle=False,
    )
    return train_ds, val_ds, train_ds.class_names


def augment_dataset(ds, augmentor, training: bool):
    AUTOTUNE = tf.data.AUTOTUNE
    # Normalise to [0,1]
    ds = ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (augmentor(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# ─── Model Builder ────────────────────────────────────────────────────────────
def build_model(num_classes: int, img_size: int, frozen: bool = True) -> tf.keras.Model:
    inputs   = tf.keras.Input(shape=(img_size, img_size, 3))
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_tensor=inputs
    )
    backbone.trainable = not frozen

    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    # float32 output even in mixed-precision mode
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax",
                                     dtype="float32")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="AgroVision_EfficientNetB0")


# ─── Callbacks ────────────────────────────────────────────────────────────────
def build_callbacks(model_out: str, log_tag: str) -> list:
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = f"logs/{log_tag}_{ts}"
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3,
            patience=4, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_out.replace(".h5", f"_{log_tag}_best.h5"),
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1),
    ]


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("  AgroVision Pro — EfficientNetB0 Training")
    print(f"{'='*60}")
    print(f"  Dataset  : {args.data_dir}")
    print(f"  Image sz : {args.img_size}×{args.img_size}")
    print(f"  Batch    : {args.batch}")
    print(f"  TL epochs: {args.epochs_tl}  |  FT epochs: {args.epochs_ft}")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds_raw, val_ds_raw, class_names = load_datasets(
        args.data_dir, args.img_size, args.batch
    )
    num_classes = len(class_names)
    print(f"  Classes  : {num_classes}  ({class_names[:5]}...)\n")

    augmentor = build_augmentation_model(args.img_size)
    train_ds  = augment_dataset(train_ds_raw, augmentor, training=True)
    val_ds    = augment_dataset(val_ds_raw,   augmentor, training=False)

    # ── Save class names ──────────────────────────────────────────────────
    with open(args.classes_out, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"  Class names saved → {args.classes_out}")

    # ══ PHASE 1 — Transfer Learning (frozen backbone) ════════════════════
    print("\n--- Phase 1: Transfer Learning (frozen backbone) ---")
    model = build_model(num_classes, args.img_size, frozen=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr_tl),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary(print_fn=lambda x: None)   # suppress noisy output

    history_tl = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_tl,
        callbacks=build_callbacks(args.model_out, "transfer"),
        verbose=1,
    )
    val_acc_tl = max(history_tl.history.get("val_accuracy", [0]))
    print(f"\n  TL Best val_accuracy: {val_acc_tl:.4f}")

    # ══ PHASE 2 — Fine-Tuning (partial unfreeze) ═════════════════════════
    print(f"\n--- Phase 2: Fine-Tuning (unfreeze from layer {args.unfreeze_at}) ---")
    backbone = model.layers[1]   # EfficientNetB0 is layer index 1
    backbone.trainable = True
    for layer in backbone.layers[:args.unfreeze_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr_ft),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_ft,
        callbacks=build_callbacks(args.model_out, "finetune"),
        verbose=1,
    )
    val_acc_ft = max(history_ft.history.get("val_accuracy", [0]))
    print(f"\n  FT Best val_accuracy: {val_acc_ft:.4f}")

    # ── Save final model ──────────────────────────────────────────────────
    model.save(args.model_out)
    print(f"\n  ✅ Model saved → {args.model_out}")
    print(f"  Best accuracy achieved: {max(val_acc_tl, val_acc_ft):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
