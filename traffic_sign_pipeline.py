from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:
    import tensorflow as tf
except ImportError:
    tf = None


IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 15
SEED = 42
N_CLASSES = 43

MODEL_DIRNAME = "artifacts"
MODEL_FILENAME = "traffic_sign_cnn.keras"
METADATA_FILENAME = "model_metadata.json"
HISTORY_FILENAME = "training_history.csv"


@dataclass(frozen=True)
class DatasetBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    meta_df: pd.DataFrame
    dataset_dir: Path


@dataclass(frozen=True)
class InferenceBundle:
    test_df: pd.DataFrame
    meta_df: pd.DataFrame
    dataset_dir: Path


@dataclass(frozen=True)
class PredictionResult:
    image: Image.Image
    predicted_class: int
    confidence: float
    probabilities: np.ndarray


def ensure_tensorflow():
    if tf is None:
        raise ImportError(
            "TensorFlow n'est pas installe. Lancez `pip install -r requirements.txt` "
            "avant d'entrainer ou d'executer l'application."
        )
    return tf


def get_project_dir(project_dir: str | Path | None = None) -> Path:
    if project_dir is None:
        return Path(__file__).resolve().parent
    return Path(project_dir).resolve()


def get_dataset_dir(project_dir: str | Path | None = None) -> Path:
    return get_project_dir(project_dir) / "GTSRB"


def get_artifacts_dir(project_dir: str | Path | None = None) -> Path:
    return get_project_dir(project_dir) / MODEL_DIRNAME


def get_model_path(project_dir: str | Path | None = None) -> Path:
    return get_artifacts_dir(project_dir) / MODEL_FILENAME


def get_metadata_path(project_dir: str | Path | None = None) -> Path:
    return get_artifacts_dir(project_dir) / METADATA_FILENAME


def get_history_path(project_dir: str | Path | None = None) -> Path:
    return get_artifacts_dir(project_dir) / HISTORY_FILENAME


def _load_csv_with_paths(csv_path: Path, dataset_dir: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    dataframe = pd.read_csv(csv_path).copy()
    if "Path" not in dataframe.columns:
        raise ValueError(f"La colonne 'Path' est absente dans {csv_path.name}.")

    dataframe["full_path"] = dataframe["Path"].map(lambda relative_path: str(dataset_dir / relative_path))
    return dataframe


def load_dataset_bundle(project_dir: str | Path | None = None) -> DatasetBundle:
    dataset_dir = get_dataset_dir(project_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dossier dataset introuvable : {dataset_dir}. "
            "Le repertoire GTSRB doit etre place a la racine du projet."
        )

    train_df = _load_csv_with_paths(dataset_dir / "Train.csv", dataset_dir)
    test_df = _load_csv_with_paths(dataset_dir / "Test.csv", dataset_dir)
    meta_df = _load_csv_with_paths(dataset_dir / "Meta.csv", dataset_dir)

    return DatasetBundle(
        train_df=train_df,
        test_df=test_df,
        meta_df=meta_df,
        dataset_dir=dataset_dir,
    )


def load_inference_bundle(project_dir: str | Path | None = None) -> InferenceBundle:
    dataset_dir = get_dataset_dir(project_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dossier dataset introuvable : {dataset_dir}. "
            "Le repertoire GTSRB doit etre place a la racine du projet."
        )

    test_df = _load_csv_with_paths(dataset_dir / "Test.csv", dataset_dir)
    meta_df = _load_csv_with_paths(dataset_dir / "Meta.csv", dataset_dir)

    return InferenceBundle(
        test_df=test_df,
        meta_df=meta_df,
        dataset_dir=dataset_dir,
    )


def split_train_validation(
    train_df: pd.DataFrame,
    validation_size: float = 0.20,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    train_split, validation_split = train_test_split(
        train_df,
        test_size=validation_size,
        random_state=seed,
        stratify=train_df["ClassId"],
    )

    return train_split.reset_index(drop=True), validation_split.reset_index(drop=True)


if tf is not None:

    class TrafficSignGenerator(tf.keras.utils.Sequence):
        def __init__(
            self,
            dataframe: pd.DataFrame,
            batch_size: int = BATCH_SIZE,
            image_size: tuple[int, int] = IMAGE_SIZE,
            shuffle: bool = True,
        ) -> None:
            self.dataframe = dataframe.reset_index(drop=True).copy()
            self.batch_size = batch_size
            self.image_size = image_size
            self.shuffle = shuffle
            self.indices = np.arange(len(self.dataframe))
            self.on_epoch_end()

        def __len__(self) -> int:
            return int(np.ceil(len(self.dataframe) / self.batch_size))

        def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
            start = index * self.batch_size
            end = min((index + 1) * self.batch_size, len(self.dataframe))
            batch_indices = self.indices[start:end]
            batch_df = self.dataframe.iloc[batch_indices]

            batch_images = np.zeros(
                (len(batch_df), self.image_size[0], self.image_size[1], 3),
                dtype=np.float32,
            )
            batch_labels = batch_df["ClassId"].astype(np.int32).to_numpy()

            for row_index, path in enumerate(batch_df["full_path"]):
                with Image.open(path) as raw_image:
                    image = raw_image.convert("RGB").resize(self.image_size)
                batch_images[row_index] = np.asarray(image, dtype=np.float32) / 255.0

            return batch_images, batch_labels

        def on_epoch_end(self) -> None:
            if self.shuffle:
                np.random.shuffle(self.indices)

else:

    class TrafficSignGenerator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            ensure_tensorflow()


def build_model(
    image_size: tuple[int, int] = IMAGE_SIZE,
    n_classes: int = N_CLASSES,
):
    ensure_tensorflow()

    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
    from tensorflow.keras.models import Sequential

    return Sequential(
        [
            Input(shape=(image_size[0], image_size[1], 3)),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.4),
            Dense(n_classes, activation="softmax"),
        ]
    )


def compile_model(model) -> None:
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def load_trained_model(
    project_dir: str | Path | None = None,
    model_path: str | Path | None = None,
):
    tensorflow_lib = ensure_tensorflow()
    resolved_model_path = Path(model_path).resolve() if model_path else get_model_path(project_dir)

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Modele introuvable : {resolved_model_path}. "
            "Lancez `python train_model.py` pour generer le fichier .keras."
        )

    return tensorflow_lib.keras.models.load_model(resolved_model_path)


def open_rgb_image(image_source: str | Path | Any) -> Image.Image:
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    with Image.open(image_source) as raw_image:
        return raw_image.convert("RGB")


def preprocess_image(
    image_source: str | Path | Any,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[Image.Image, np.ndarray]:
    image = open_rgb_image(image_source)
    image_resized = image.resize(image_size)
    image_array = np.asarray(image_resized, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image, image_array


def predict_image(
    model,
    image_source: str | Path | Any,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> PredictionResult:
    image, image_array = preprocess_image(image_source, image_size=image_size)
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    return PredictionResult(
        image=image,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
    )


def top_predictions(probabilities: np.ndarray, top_k: int = 3) -> list[dict[str, float | int]]:
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    return [
        {
            "ClassId": int(class_index),
            "Confiance": float(probabilities[class_index]),
        }
        for class_index in top_indices
    ]


def get_reference_image_path(meta_df: pd.DataFrame, class_id: int) -> Path | None:
    reference_rows = meta_df.loc[meta_df["ClassId"] == class_id, "full_path"]
    if reference_rows.empty:
        return None
    return Path(reference_rows.iloc[0])


def save_training_metadata(
    metadata: dict[str, Any],
    project_dir: str | Path | None = None,
) -> Path:
    metadata_path = get_metadata_path(project_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata_path


def load_training_metadata(project_dir: str | Path | None = None) -> dict[str, Any]:
    metadata_path = get_metadata_path(project_dir)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))
