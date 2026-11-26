import json, os, shutil, datetime, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(os.getenv("ECOSORT_PROJECT_DIR", Path(__file__).resolve().parent.parent))
MODELS = ROOT / "models"
SPLITS = ROOT / "splits"
REPORTS = ROOT / "reports"
INBOX = ROOT / "data_inbox"
META_PATH = MODELS / "model_metadata.json"
METRICS_PATH = REPORTS / "metrics.json"
SEED = 42
BATCH = 32
IMG_SIZE = 224
EPOCHS = 5
LR = 1e-5
AUTOTUNE = tf.data.AUTOTUNE

def find_realwaste_root():
    raw = ROOT / "data_raw"
    for p in raw.rglob("RealWaste"):
        if p.is_dir() and any(x.is_dir() for x in p.iterdir()):
            return p
    raise FileNotFoundError("Could not locate data_raw/**/RealWaste class folders.")

def make_ds(frame, root, training=False):
    paths = [str(Path(root) / p) for p in frame["path"]]
    labels = frame["label"].astype("int32").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    def load(path, label):
        raw = tf.io.read_file(path)
        image = tf.image.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return image, label
    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if training:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.1),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)

def main():
    tf.keras.utils.set_random_seed(SEED)
    meta = json.loads(META_PATH.read_text())
    classes = meta["class_names"]
    active_path = MODELS / meta["model_file"]
    if not active_path.exists():
        raise FileNotFoundError(active_path)

    train = pd.read_csv(SPLITS / "train.csv")
    val = pd.read_csv(SPLITS / "val.csv")
    data_root = find_realwaste_root()

    # Add only staged, labelled images to the training frame.
    added = []
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    for class_name in classes:
        folder = INBOX / class_name
        if not folder.exists():
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in allowed_ext:
                added.append({
                    "path": str(image_path.resolve()),
                    "label": classes.index(class_name)
                })

    if not added:
        raise RuntimeError(
            "No labelled images found in data_inbox/<class_name>/. "
            "Upload a ZIP containing one folder per class first."
        )

    # Original split paths are relative to the RealWaste image root;
    # staged upload paths are absolute and remain readable by Path(root) / path.
    extra = pd.DataFrame(added)
    combined = pd.concat([train[["path", "label"]], extra], ignore_index=True)
    # Recalculate balanced class weights after adding the labelled examples.
    counts = np.bincount(combined["label"].astype(int), minlength=len(classes))
    class_weights = {
        i: float(len(combined) / (len(classes) * counts[i]))
        for i in range(len(classes)) if counts[i] > 0
    }

    train_ds = make_ds(combined, data_root, training=True)
    val_ds = make_ds(val[["path", "label"]], data_root, training=False)

    active_model = tf.keras.models.load_model(active_path, compile=False)
    active_model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    baseline_loss, baseline_acc = active_model.evaluate(val_ds, verbose=0)

    # Find the nested MobileNetV2 model and fine-tune its final layers only.
    nested_models = [layer for layer in active_model.layers
                     if isinstance(layer, tf.keras.Model)]
    if not nested_models:
        raise RuntimeError("Could not find the nested MobileNetV2 base model.")
    base = nested_models[0]
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    active_model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7
        )
    ]
    history = active_model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        class_weight=class_weights, callbacks=callbacks, verbose=1
    )
    candidate_loss, candidate_acc = active_model.evaluate(val_ds, verbose=0)

    # Keep candidate artifacts separate until the gate has passed.
    version_number = 1
    import re
    found = re.search(r"v(\d+)$", str(meta.get("model_version", "v1")))
    if found:
        version_number = int(found.group(1)) + 1
    candidate_version = f"v{version_number}"
    candidate_name = f"ecosort_mobilenetv2_{candidate_version}.keras"
    candidate_path = MODELS / candidate_name
    active_model.save(candidate_path)

    # Conservative, validation-only promotion rule.
    promoted = (
        candidate_loss < float(baseline_loss)
        and candidate_acc >= float(baseline_acc) - 0.01
    )

    result = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "previous_version": meta.get("model_version"),
        "candidate_version": candidate_version,
        "uploaded_images_used": len(added),
        "training_images_total": len(combined),
        "baseline_validation_loss": float(baseline_loss),
        "baseline_validation_accuracy": float(baseline_acc),
        "candidate_validation_loss": float(candidate_loss),
        "candidate_validation_accuracy": float(candidate_acc),
        "promotion_gate": "candidate val_loss lower AND candidate val_accuracy >= baseline - 0.01",
        "promoted": bool(promoted),
        "epochs_completed": len(history.history.get("loss", [])),
        "test_set_used": False
    }

    if promoted:
        new_meta = dict(meta)
        new_meta["model_version"] = candidate_version
        new_meta["model_file"] = candidate_name
        new_meta["created_utc"] = result["created_utc"]
        new_meta["last_retraining"] = result
        # Historical test metrics belong to the original evaluated model.
        new_meta["test_metrics_note"] = (
            "Test metrics in this metadata refer to the original independently "
            "evaluated model; the new version has not been evaluated on the test set."
        )
        tmp = META_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(new_meta, indent=2))
        os.replace(tmp, META_PATH)
        result["status"] = "promoted"
    else:
        candidate_path.unlink(missing_ok=True)
        result["status"] = "not_promoted"
        result["reason"] = "Candidate did not pass the validation-only promotion gate."

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "last_retraining.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    main()
