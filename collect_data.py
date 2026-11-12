"""Record Xbox-controlled SO-101 demonstrations."""

import ctypes
from datetime import datetime
from pathlib import Path
from random import choice, randrange
from time import perf_counter

import cv2
import numpy as np

from controller_demo import ControllerState, GripperIK, READY_POSE, move_target, read_controller
from cube_sim import make_sim

PROMPTS = [
    "Pick up the blue cube",
    "Pick up the green cube",
    "Pick up the yellow cube",
]
OUTPUT = Path(__file__).with_name("recordings")
WINDOW = "SO-101 | Collect data"
CAMERAS = ("top", "wrist")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class Episode:
    def __init__(self, prompt, seed, fps):
        self.prompt, self.seed, self.fps = prompt, seed, fps
        self.states, self.actions = [], []
        self.videos = {}
        self.folder = OUTPUT / datetime.now().strftime("episode_%Y%m%d_%H%M%S_%f")

    def add(self, obs, action):
        if not self.videos:
            self.folder.mkdir(parents=True)
            for name in CAMERAS:
                height, width = obs["pixels"][name].shape[:2]
                video = cv2.VideoWriter(
                    str(self.folder / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
                )
                self.videos[name] = video
                if not video.isOpened():
                    raise RuntimeError(f"Could not open video writer for {name} in {self.folder}")

        for name, video in self.videos.items():
            video.write(cv2.cvtColor(obs["pixels"][name], cv2.COLOR_RGB2BGR))
        self.states.append(obs["agent_pos"].copy())
        self.actions.append(action.copy())

    def save(self):
        for video in self.videos.values():
            video.release()
        if self.actions:
            np.savez(
                self.folder / "episode.npz",
                **{
                    "observation.state": np.asarray(self.states, dtype=np.float32),
                    "action": np.asarray(self.actions, dtype=np.float32),
                    "timestamp": np.arange(len(self.actions), dtype=np.float64) / self.fps,
                    "task": self.prompt,
                    "fps": self.fps,
                    "seed": self.seed,
                    "joint_names": JOINTS,
                    "joint_units": ("degrees",) * 5 + ("0-100",),
                },
            )
            print(f"Saved {len(self.actions)} frames to {self.folder}", flush=True)


def collect(env, get_state):
    env.initial_joints[:] = READY_POSE
    obs, _ = env.reset(seed=0)
    ik = GripperIK(env)
    episode = None
    dt = 1 / env.fps
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 480)

    try:
        while True:
            start = perf_counter()
            gamepad = read_controller(get_state)
            status = "Press R to start | Q to quit"
            if episode is not None:
                status = f"{episode.prompt} | Recording"
                if gamepad is None:
                    status = f"{episode.prompt} | Connect controller to record"
                else:
                    target, roll, gripper = move_target(gamepad, target, roll, dt)
                    roll = np.clip(roll, ik.world.lo[4], ik.world.hi[4])
                    action = ik.command(target, action, roll, gripper)
                    next_obs, *_ = env.step(action)
                    # Pair the command with the observation from before the step.
                    episode.add(obs, action)
                    obs = next_obs

            pixels = obs["pixels"]
            preview = cv2.cvtColor(np.hstack([pixels[name] for name in CAMERAS]), cv2.COLOR_RGB2BGR)
            cv2.putText(preview, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow(WINDOW, preview)
            delay_ms = max(1, round(1000 * (dt - (perf_counter() - start))))
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("r"), ord("R")):
                if episode is not None:
                    previous, episode = episode, None
                    previous.save()
                prompt, seed = choice(PROMPTS), randrange(2**32)
                obs, _ = env.reset(seed=seed)
                action = obs["agent_pos"]
                target = ik.tip()
                roll = np.deg2rad(action[4])
                episode = Episode(prompt, seed, env.fps)
    finally:
        if episode is not None:
            episode.save()


def main():
    get_state = ctypes.WinDLL("xinput1_4").XInputGetState
    get_state.argtypes = (ctypes.c_uint32, ctypes.POINTER(ControllerState))
    get_state.restype = ctypes.c_uint32
    try:
        with make_sim(workspace_camera="top", include_red_cube=False) as env:
            collect(env, get_state)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
