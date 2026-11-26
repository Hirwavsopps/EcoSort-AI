import json
from pathlib import Path
from typing import Any
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

class EcoSortPredictor:
    def __init__(self, model_path: str, metadata_path: str, rules_path: str | None = None):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.metadata = json.loads(self.metadata_path.read_text())
        self.model = tf.keras.models.load_model(self.model_path)
        self.class_names = self.metadata["class_names"]
        self.image_size = tuple(self.metadata.get("input_size", [224, 224, 3])[:2])
        self.threshold = float(self.metadata.get("confidence_threshold", 0.70))
        self.rules = {}
        if rules_path and Path(rules_path).exists():
            self.rules = json.loads(Path(rules_path).read_text())

    def prepare_image(self, image_file):
        try:
            with Image.open(image_file) as image:
                image = image.convert("RGB").resize(
                    (self.image_size[1], self.image_size[0])
                )
                array = np.asarray(image, dtype=np.float32)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("File is not a valid readable image.") from exc
        return np.expand_dims(array, axis=0)

    def predict(self, image_file, top_k: int = 3) -> dict[str, Any]:
        if not 1 <= top_k <= len(self.class_names):
            raise ValueError(f"top_k must be between 1 and {len(self.class_names)}")
        batch = self.prepare_image(image_file)
        probabilities = self.model.predict(batch, verbose=0)[0]
        indices = np.argsort(probabilities)[::-1][:top_k]
        top = [
            {"class_name": self.class_names[int(i)],
             "confidence": round(float(probabilities[i]), 4)}
            for i in indices
        ]
        best = top[0]
        uncertain = best["confidence"] < self.threshold
        stream = None
        if not uncertain:
            stream = self.rules.get("streams", {}).get(best["class_name"])
        manual_review = (
            uncertain or best["class_name"] in self.rules.get("manual_review_classes", [])
        )
        return {
            "predicted_class": best["class_name"],
            "confidence": best["confidence"],
            "uncertain": uncertain,
            "top_k": top,
            "disposal_stream": None if manual_review else stream,
            "manual_review": manual_review,
            "model_version": self.metadata["model_version"]
        }
