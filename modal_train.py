"""Run train.py on a Modal GPU and persist versioned model outputs."""

import os
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent
LOCAL_DATASET = PROJECT_ROOT.parent / "source_dataset"
REMOTE_DATASET = "/data/source_dataset"
REMOTE_MODELS = "/models"
AWS_ROLE_ARN = "arn:aws:iam::832271495954:role/CelebrityTwinModalTrainingRole"

app = modal.App("celebrity-doppelganger-training")
model_volume = modal.Volume.from_name("celebrity-doppelganger-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "facenet-pytorch>=2.6.0,<3",
        "boto3>=1.35,<2",
        "lancedb==0.21.2",
        "numpy>=1.26,<3",
        "pandas>=2.2,<3",
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


def configure_aws_identity() -> None:
    import boto3

    credentials = boto3.client("sts").assume_role_with_web_identity(
        RoleArn=os.environ["AWS_ROLE_ARN"],
        RoleSessionName="celebrity-twin-modal-training",
        WebIdentityToken=os.environ["MODAL_IDENTITY_TOKEN"],
    )["Credentials"]
    os.environ.update(
        {
            "AWS_ACCESS_KEY_ID": credentials["AccessKeyId"],
            "AWS_SECRET_ACCESS_KEY": credentials["SecretAccessKey"],
            "AWS_SESSION_TOKEN": credentials["SessionToken"],
        }
    )


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 12,
    volumes={REMOTE_MODELS: model_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    env={
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_ROLE_ARN": AWS_ROLE_ARN,
        "PROTOTYPE_DATABASE_URI": "s3://celebrity-twin-832271495954-us-east-1/vector/prototypes.lancedb",
        "CELEBRITY_PHOTO_S3_PREFIX": "s3://celebrity-twin-832271495954-us-east-1/photos/celebrities",
        "MODEL_CHECKPOINT_S3_URI": "s3://celebrity-twin-832271495954-us-east-1/models/celebrity_face_classifier.pt",
        "WANDB_ENTITY": "gabomfim-unicamp",
    },
)
def train_remote(
    epochs: int = 30,
    run_name: str | None = None,
    prototypes_only: bool = False,
    full_data: bool = False,
) -> str:
    import sys

    import boto3

    configure_aws_identity()
    sys.path.insert(0, "/app")
    from build_prototypes import build_prototype_database
    from train import TrainConfig, train

    run_output = Path(REMOTE_MODELS) / (run_name or "latest")
    final_path = run_output / "celebrity_face_classifier.pt"
    if prototypes_only:
        if not final_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found in Modal Volume: {final_path}")
    else:
        final_path = train(
            TrainConfig(
                data_dir=REMOTE_DATASET,
                output_dir=str(run_output),
                epochs=epochs,
                num_workers=8,
                run_name=run_name,
                full_data=full_data,
            )
        )
    build_prototype_database(
        final_path,
        Path(REMOTE_DATASET),
        os.environ["PROTOTYPE_DATABASE_URI"],
        run_output / "person_metadata.json",
        os.environ.get("CELEBRITY_PHOTO_S3_PREFIX"),
        num_workers=8,
    )
    boto3.client("s3").upload_file(
        str(final_path),
        "celebrity-twin-832271495954-us-east-1",
        "models/celebrity_face_classifier.pt",
    )
    model_volume.commit()
    return str(final_path)


@app.local_entrypoint()
def main(
    epochs: int = 30,
    run_name: str | None = None,
    prototypes_only: bool = False,
    full_data: bool = False,
) -> None:
    result = train_remote.remote(epochs, run_name, prototypes_only, full_data)
    print(f"Final model saved in Modal Volume: {result}")
