"""Move the simulated arm and show its overhead and wrist cameras."""

import argparse
import numpy as np
import cv2

from cube_sim import make_sim

WINDOW = "SO-101 | Overhead and wrist cameras"
RENDER = True

# CAMERA VIEWS:
# Wrist cam
# Top Camera: 48cm forward, 58cm up, 25.77 degrees tilt from verticle, 60 degree verticle field of view

def main():
    with make_sim(workspace_camera="top") as env:
        obs, _ = env.reset(seed=0)
        home = obs["agent_pos"].copy()
        action = home.copy()

        print(env.task_description)
        print("Camera shapes:", {name: image.shape for name, image in obs["pixels"].items()})
        print("Starting joint positions:", home.round(1))

        print("Press Q or Esc to close the camera window.")

        for step in range(300):
            action[4] = -90
            obs, _, terminated, truncated, info = env.step(action)

            if RENDER:
                pixels = obs["pixels"]
                cameras = np.hstack((pixels["top"], pixels["wrist"]))
                cv2.imshow(WINDOW, cv2.cvtColor(cameras, cv2.COLOR_RGB2BGR))
                if (cv2.waitKey(33) & 0xFF) in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break

            if terminated or truncated:
                break

        print("Ending joint positions:", obs["agent_pos"].round(1))
        print("Cubes in box:", info["cubes_in_box"])

    if RENDER:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
