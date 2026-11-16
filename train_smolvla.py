"""Fine-tune SmolVLA on the local dataset created by prepare_dataset.py."""

import argparse
from datetime import datetime
from importlib.util import find_spec
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
CAMERA_MAP = {"observation.images.top": "observation.images.camera1", 
              "observation.images.wrist": "observation.images.camera2"}

def fix_smolvla_padding():
    """Supply the padding key expected by LeRobot 0.4.4."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    original_forward = SmolVLAPolicy.forward

    def forward(policy, batch, *args, **kwargs):
        if "action_is_pad" in batch and "actions_id_pad" not in batch:
            batch = {**batch, "actions_id_pad": batch["action_is_pad"]}
        return original_forward(policy, batch, *args, **kwargs)

    SmolVLAPolicy.forward = forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "datasets" / "cube_pickup")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Start a new run from a numbered checkpoint or its pretrained_model folder",
    )
    parser.add_argument("--steps", type=int, default=15_000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    if not (dataset / "meta" / "info.json").is_file():
        parser.error(f"No LeRobotDataset at {dataset}. Run prepare_dataset.py first.")
    if min(args.steps, args.batch_size) < 1:
        parser.error("--steps and --batch-size must be positive.")
    model_path = args.checkpoint
    if model_path is not None:
        model_path = model_path.resolve()
        if (model_path / "pretrained_model").is_dir():
            model_path /= "pretrained_model"
        for filename in ("config.json", "model.safetensors"):
            if not (model_path / filename).is_file():
                parser.error(f"No {filename} found in {model_path}.")

    if find_spec("lerobot") is None:
        parser.error("Use the Python environment where LeRobot is installed.")

    import torch

    if not torch.cuda.is_available():
        parser.error("This Python needs CUDA-enabled PyTorch for GPU training. See README.md.")
    if any(find_spec(package) is None for package in ("transformers", "num2words", "safetensors")):
        parser.error("Install the LeRobot SmolVLA dependencies in this environment. See README.md.")

    output = args.output or PROJECT / "outputs" / datetime.now().strftime("smolvla_%Y%m%d_%H%M%S")
    print(f"Training on {torch.cuda.get_device_name()} | Checkpoints: {output.resolve()}", flush=True)
    if model_path is None:
        from huggingface_hub import snapshot_download

        # A local path avoids LeRobot treating the Hub ID as a Windows path.
        model_path = snapshot_download("lerobot/smolvla_base", allow_patterns=["*.json", "*.safetensors"])

    print(f"Starting from {model_path}", flush=True)
    sys.argv = [
        sys.argv[0],
        f"--policy.path={model_path}",
        "--resume=false",
        "--policy.device=cuda",
        "--policy.train_expert_only=false",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=local/cube_pickup",
        f"--dataset.root={dataset}",
        "--dataset.video_backend=pyav",
        f"--rename_map={json.dumps(CAMERA_MAP)}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        "--num_workers=0",
        "--save_freq=1000",
        "--wandb.enable=false",
        f"--output_dir={output.resolve()}",
    ]
    from lerobot.scripts import lerobot_train

    if sys.platform == "win32":
        update_last_checkpoint = lerobot_train.update_last_checkpoint

        def update_last_if_allowed(checkpoint_dir):
            try:
                update_last_checkpoint(checkpoint_dir)
            except OSError as exc:
                if exc.winerror != 1314:
                    raise
                print(f"Windows skipped the 'last' shortcut. Checkpoint saved to {checkpoint_dir}.", flush=True)

        lerobot_train.update_last_checkpoint = update_last_if_allowed

    fix_smolvla_padding()
    lerobot_train.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
