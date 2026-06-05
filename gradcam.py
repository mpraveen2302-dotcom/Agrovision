"""
gradcam.py — AgroVision Pro
Grad-CAM Explainable AI
Highlights disease hotspots on the leaf image
"""

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _find_last_conv_layer(model: tf.keras.Model) -> str:
    """Return the name of the last Conv2D layer in the model."""
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            last_conv = layer.name
        # Handle nested models (e.g. EfficientNet / MobileNet backbone)
        elif hasattr(layer, "layers"):
            for sub in layer.layers:
                if isinstance(sub, (tf.keras.layers.Conv2D,
                                     tf.keras.layers.DepthwiseConv2D)):
                    last_conv = sub.name
    if last_conv is None:
        raise ValueError("No Conv2D layer found in the model.")
    return last_conv


def _make_gradcam_heatmap(img_array: np.ndarray,
                           model: tf.keras.Model,
                           last_conv_layer_name: str,
                           pred_index: int | None = None) -> np.ndarray:
    """
    Core Grad-CAM computation.

    Parameters
    ----------
    img_array           : (1, H, W, 3) float32 array, values 0–1
    model               : Keras model
    last_conv_layer_name: name of last conv layer
    pred_index          : class index; if None uses argmax

    Returns
    -------
    heatmap : (H', W') float32 array, values 0–1
    """
    # Build sub-model: input → [conv_output, predictions]
    grad_model = tf.keras.models.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(last_conv_layer_name).output,
                   model.output]
    )

    with tf.GradientTape() as tape:
        inputs      = tf.cast(img_array, tf.float32)
        conv_output, predictions = grad_model(inputs)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads     = tape.gradient(class_channel, conv_output)
    pooled    = tf.reduce_mean(grads, axis=(0, 1, 2))        # global avg pool
    conv_out  = conv_output[0]
    heatmap   = conv_out @ pooled[..., tf.newaxis]
    heatmap   = tf.squeeze(heatmap)
    heatmap   = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _overlay_heatmap(pil_image: Image.Image,
                     heatmap: np.ndarray,
                     alpha: float = 0.45,
                     colormap: int = cv2.COLORMAP_JET) -> Image.Image:
    """
    Overlay the Grad-CAM heatmap onto the original image.

    Returns a PIL Image (RGB).
    """
    img_np  = np.array(pil_image.convert("RGB"))
    h, w    = img_np.shape[:2]

    # Resize heatmap to image size
    heat_u8 = np.uint8(255 * heatmap)
    heat_u8 = cv2.resize(heat_u8, (w, h))
    colored = cv2.applyColorMap(heat_u8, colormap)           # BGR
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(img_np, 1 - alpha, colored, alpha, 0)
    return Image.fromarray(blended)


# ─── Public API ───────────────────────────────────────────────────────────────
def generate_gradcam(pil_image: Image.Image,
                     model: tf.keras.Model,
                     pred_index: int | None = None,
                     target_size: tuple[int, int] = (224, 224),
                     alpha: float = 0.45) -> dict:
    """
    Generate a Grad-CAM overlay for a leaf image.

    Parameters
    ----------
    pil_image   : PIL Image (RGB)
    model       : loaded Keras model
    pred_index  : class index to explain; None → top-1
    target_size : (H, W) expected by the model
    alpha       : heatmap blend strength (0–1)

    Returns
    -------
    dict:
        overlay_image   : PIL Image  (original + heatmap overlay)
        heatmap_image   : PIL Image  (pure coloured heatmap)
        heatmap_array   : np.ndarray (H, W) float32, 0–1
        pred_index      : int
        hotspot_pct     : float  (% of image with high activation ≥ 0.5)
        conv_layer      : str    (layer name used)
        error           : str or None
    """
    try:
        # Find last conv layer
        conv_layer = _find_last_conv_layer(model)

        # Preprocess
        img_resized = pil_image.convert("RGB").resize(target_size[::-1])  # PIL: (W,H)
        img_array   = np.expand_dims(np.array(img_resized) / 255.0, axis=0).astype(np.float32)

        # Compute heatmap
        heatmap = _make_gradcam_heatmap(img_array, model, conv_layer, pred_index)

        # Derive pred_index if not provided
        if pred_index is None:
            preds      = model.predict(img_array, verbose=0)[0]
            pred_index = int(np.argmax(preds))

        # Overlay on *original* (not resized) image
        overlay   = _overlay_heatmap(pil_image, heatmap, alpha=alpha)

        # Pure heatmap image
        heat_u8   = np.uint8(255 * cv2.resize(heatmap,
                             (pil_image.width, pil_image.height)))
        colored   = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_pil  = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))

        # Hotspot percentage
        hotspot_pct = round(float(np.mean(heatmap >= 0.5)) * 100, 1)

        return dict(
            overlay_image = overlay,
            heatmap_image = heat_pil,
            heatmap_array = heatmap,
            pred_index    = pred_index,
            hotspot_pct   = hotspot_pct,
            conv_layer    = conv_layer,
            error         = None,
        )

    except Exception as exc:
        return dict(
            overlay_image = pil_image,
            heatmap_image = pil_image,
            heatmap_array = np.zeros((7, 7), dtype=np.float32),
            pred_index    = pred_index or 0,
            hotspot_pct   = 0.0,
            conv_layer    = "unknown",
            error         = str(exc),
        )
