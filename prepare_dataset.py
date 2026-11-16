"""Convert saved controller demonstrations to a local LeRobotDataset."""

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent
REPO_ID = "local/cube_pickup"
CAMERAS = ("top", "wrist")
STATE = "observation.state"


def read_episode(archive):
    with np.load(archive, allow_pickle=False) as data:
        state = data[STATE]
        action = data["action"]
        task = str(data["task"].item())
        fps = int(data["fps"].item())
        names = data["joint_names"].tolist()
        units = data["joint_units"].tolist()
        timestamps = data["timestamp"]

    if (state.ndim != 2 or state.shape[1] != 6 or not len(state) or action.shape != state.shape
            or not np.isfinite(state).all() or not np.isfinite(action).all()):
        raise ValueError(f"{archive}: expected matching finite (frames, 6) state and action arrays.")
    
    if fps <= 0 or not task.strip() or len(names) != 6 or units != ["degrees"] * 5 + ["0-100"]:
        raise ValueError(f"{archive}: invalid task, frame rate, or joint metadata.")
    
    if timestamps.shape != (len(state),) or not np.allclose(timestamps, np.arange(len(state)) / fps):
        raise ValueError(f"{archive}: timestamps do not match the saved frame rate.")
    
    return state.astype(np.float32), action.astype(np.float32), task, fps, names


def convert(archives, output):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    counts = Counter()
    dataset = None
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the output folder only after all recordings have passed validation.
    with TemporaryDirectory(prefix=".convert-", dir=output.parent) as temp:
        staged = Path(temp) / "dataset"
        try:
            for archive in archives:
                states, actions, task, fps, names = read_episode(archive)
                videos = [cv2.VideoCapture(str(archive.parent / f"{camera}.mp4")) for camera in CAMERAS]
                try:
                    if not all(video.isOpened() for video in videos):
                        raise ValueError(f"{archive.parent}: cannot open both camera videos.")
                    for i, (state, action) in enumerate(zip(states, actions)):
                        frame = {STATE: state, "action": action, "task": task}
                        for camera, video in zip(CAMERAS, videos):
                            ok, bgr = video.read()
                            if not ok:
                                raise ValueError(f"{archive.parent}: {camera}.mp4 is missing frame {i}.")
                            frame[f"observation.images.{camera}"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        if dataset is None:
                            joints = {"dtype": "float32", "shape": (6,), "names": names}
                            features = {STATE: joints.copy(), "action": joints.copy()}
                            for camera in CAMERAS:
                                key = f"observation.images.{camera}"
                                features[key] = {
                                    "dtype": "video",
                                    "shape": frame[key].shape,
                                    "names": ["height", "width", "channels"],
                                }
                            dataset = LeRobotDataset.create(
                                repo_id=REPO_ID, root=staged, fps=fps, features=features,
                                robot_type="so101_follower", vcodec="h264",
                            )
                        if i == 0 and (dataset.fps != fps or dataset.features[STATE]["names"] != names):
                            raise ValueError(f"{archive}: frame rate or joint order differs across episodes.")
                        dataset.add_frame(frame)
                    for camera, video in zip(CAMERAS, videos):
                        if video.read()[0]:
                            raise ValueError(f"{archive.parent}: {camera}.mp4 has extra frames.")
                finally:
                    for video in videos:
                        video.release()
                dataset.save_episode(parallel_encoding=False)
                counts[task] += 1
                print(f"Converted {archive.parent.name} | {task} | {len(states)} frames", flush=True)
        finally:
            if dataset is not None:
                dataset.finalize()
        staged.rename(output)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=PROJECT / "recordings")
    parser.add_argument("--output", type=Path, default=PROJECT / "datasets" / "cube_pickup")
    args = parser.parse_args()

    output = args.output.resolve()
    archives = sorted(args.recordings.glob("episode_*/episode.npz"))
    if not archives:
        parser.error(f"No saved episodes found in {args.recordings}.")
    if output.exists():
        parser.error(f"{output} already exists. Choose a new path with --output.")
    try:
        counts = convert(archives, output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")
    print(f"Saved {sum(counts.values())} episodes to {output}")
    for task, count in sorted(counts.items()):
        print(f"  {task}: {count}")


if __name__ == "__main__":
    main()
