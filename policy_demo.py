"""Run a trained SmolVLA checkpoint in the cube pickup simulator."""

import argparse
from pathlib import Path
import re
from time import perf_counter

import cv2
import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.control_utils import predict_action

from controller_demo import READY_POSE
from cube_sim import make_sim

WINDOW = "SO-101 | SmolVLA"


def requested_cube_in_box(env, cube_name):
    """Check the target is on the box floor and is the only cube inside."""
    world = env._world
    if world.cubes_in_box() != 1:
        return False

    cube = world.data.body(cube_name)
    box = world.data.body("box")
    # The first geom on the box body is its floor in the pinned scene.
    floor = world.model.body("box").geomadr[0]
    for contact in world.data.contact:
        if floor in (contact.geom1, contact.geom2):
            other = contact.geom2 if contact.geom1 == floor else contact.geom1
            if world.model.geom_bodyid[other] == cube.id:
                local = (cube.xpos - box.xpos) @ box.xmat.reshape(3, 3)
                return bool((np.abs(local[:2]) < 0.045).all() and 0 < local[2] < 0.06)
    return False


def run_demo(env, policy, preprocessor, postprocessor, prompt, cube_name):
    env.initial_joints[:] = READY_POSE
    obs, _ = env.reset()
    policy.reset()
    device = torch.device(policy.config.device)
    result = None
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 480)

    while True:
        start = perf_counter()
        if result is None:
            # The saved processors handle camera names and undo action normalization.
            observation = {
                "observation.state": obs["agent_pos"],
                "observation.images.top": obs["pixels"]["top"],
                "observation.images.wrist": obs["pixels"]["wrist"],
            }
            action = predict_action(
                observation, policy, device, preprocessor, postprocessor,
                use_amp=policy.config.use_amp, task=prompt, robot_type="so101_follower",
            ).squeeze(0).cpu().numpy()
            if action.shape != (6,) or not np.isfinite(action).all():
                raise ValueError(f"Expected six finite joint commands, got {action}.")
            obs, _, terminated, truncated, _ = env.step(action)
            if requested_cube_in_box(env, cube_name):
                result = "Success"
            elif terminated or truncated:
                result = "Episode ended without success"

        pixels = obs["pixels"]
        preview = cv2.cvtColor(np.hstack((pixels["top"], pixels["wrist"])), cv2.COLOR_RGB2BGR)
        cv2.putText(preview, prompt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        status = f"{result} | R: new layout | Q: quit" if result else "R: new layout | Q: quit"
        cv2.putText(preview, status, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(WINDOW, preview)
        delay_ms = max(1, round(1000 * (1 / env.fps - (perf_counter() - start))))
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("r"), ord("R")):
            obs, _ = env.reset()
            policy.reset()
            result = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Numbered checkpoint or its pretrained_model folder")
    parser.add_argument("prompt", help="The task instruction, in quotes")
    args = parser.parse_args()
    colors = set(re.findall(r"\b(blue|green|yellow)\s+cube\b", args.prompt.lower()))
    if len(colors) != 1:
        parser.error("Name one target cube in the prompt: blue cube, green cube, or yellow cube.")
    cube_name = f"{colors.pop()}_cube"
    checkpoint = args.checkpoint.resolve()
    if (checkpoint / "pretrained_model").is_dir():
        checkpoint /= "pretrained_model"
    if not (checkpoint / "model.safetensors").is_file():
        parser.error(f"No model.safetensors found in {checkpoint}.")

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.config.n_action_steps = 15  # Replan after ten executed actions.
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": policy.config.device}},
    )
    try:
        with make_sim(instruction=args.prompt, workspace_camera="top", include_red_cube=False) as env:
            run_demo(env, policy, preprocessor, postprocessor, args.prompt, cube_name)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
