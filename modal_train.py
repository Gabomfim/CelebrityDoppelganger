"""Run train.py on a Modal GPU and persist versioned model outputs."""

import os
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent
LOCAL_DATASET = PROJECT_ROOT.parent / "source_dataset"
REMOTE_DATASET = "/data/source_dataset"
REMOTE_MODELS = "/models"

app = modal.App("celebrity-doppelganger-training")
model_volume = modal.Volume.from_name("celebrity-doppelganger-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "facenet-pytorch>=2.6.0,<3",
        "lancedb==0.21.2",
        "numpy>=1.26,<3",
        "pillow>=10,<12",
        "pyarrow>=18,<23",
        "requests>=2.32,<3",
        "torch>=2.2,<3",
        "torchvision>=0.17,<1",
        "wandb>=0.19,<1",
    )
    .add_local_file(PROJECT_ROOT / "train.py", "/app/train.py")
    .add_local_file(PROJECT_ROOT / "build_prototypes.py", "/app/build_prototypes.py")
    .add_local_dir(LOCAL_DATASET, REMOTE_DATASET)
)


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 12,
    volumes={REMOTE_MODELS: model_volume},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name(
            "aws-secret", required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        ),
    ],
)
def train_remote(epochs: int = 30, run_name: str | None = None) -> str:
    import sys

    sys.path.insert(0, "/app")
    from build_prototypes import build_prototype_database
    from train import TrainConfig, train

    run_output = Path(REMOTE_MODELS) / (run_name or "latest")
    final_path = train(
        TrainConfig(
            data_dir=REMOTE_DATASET,
            output_dir=str(run_output),
            epochs=epochs,
            num_workers=8,
            run_name=run_name,
        )
    )
    build_prototype_database(
        final_path,
        Path(REMOTE_DATASET),
        os.environ["PROTOTYPE_DATABASE_URI"],
        run_output / "person_metadata.json",
        num_workers=8,
    )
    model_volume.commit()
    return str(final_path)


@app.local_entrypoint()
def main(epochs: int = 30, run_name: str | None = None) -> None:
    print(f"Final model saved in Modal Volume: {train_remote.remote(epochs, run_name)}")
