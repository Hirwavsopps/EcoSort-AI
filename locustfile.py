
import csv
import random
from pathlib import Path

from locust import HttpUser, task, between

PROJECT_ROOT = Path("/app")
DATASET_ROOT = (
    PROJECT_ROOT / "data_raw" / "realwaste-main" / "RealWaste"
)
TEST_CSV = PROJECT_ROOT / "splits" / "test.csv"


def load_test_images():
    images = []

    with TEST_CSV.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_path = DATASET_ROOT / row["path"]

            if image_path.is_file():
                images.append(image_path)

    if not images:
        raise RuntimeError(
            "No test images found. Check the dataset and test.csv paths."
        )

    return images


TEST_IMAGES = load_test_images()


class EcoSortUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_waste(self):
        image_path = random.choice(TEST_IMAGES)

        try:
            with image_path.open("rb") as image:
                response = self.client.post(
                    "/predict",
                    files={
                        "file": (
                            image_path.name,
                            image,
                            "image/jpeg",
                        )
                    },
                    name="/predict",
                )

            if response.status_code != 200:
                response.failure(
                    f"Prediction failed: HTTP {response.status_code}"
                )

        except OSError as error:
            print(f"Could not read image: {error}")
