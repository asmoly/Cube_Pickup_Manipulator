# SO-101 cube pickup simulator

Run Hugging Face's [SO-101 tabletop scene](https://huggingface.co/lerobot/so101-tabletop-mujoco) locally in MuJoCo. The scene includes four colored cubes, a box, and front, overhead, and wrist cameras.

## Run the demo on Windows

From PowerShell in this folder, with Python 3.10 or newer:

```powershell
python -m pip install mujoco==3.3.7 gymnasium==1.2.3 numpy huggingface_hub==2.4.0 opencv-contrib-python
python demo.py
```

A fresh setup uses the Hugging Face Hub package to download the pinned scene and robot assets into `.sim-cache`. Once the assets are cached, the demo loads them locally without importing the Hub package. The demo moves the wrist and shows the overhead and wrist views. Press **Q** or **Esc** to close it.

## Xbox controller demo

Plug in an Xbox controller and run `python demo_controller.py`.

- Left stick moves the gripper across the table at its current height.
- Right stick up/down changes height. Right stick left/right rotates the wrist.
- Right trigger closes the gripper in proportion to how far you squeeze it.
- Press **R** in the preview to reset or **Q/Esc** to quit.

## Collect demonstrations

Run `python collect_data.py` with the Xbox controller connected. Press **R** to reset the scene and start recording a randomly selected blue, green, or yellow cube prompt. Press **R** again to save and start a new episode, or **Q** to save and quit.

Each episode goes in `recordings/` with `top.mp4`, `wrist.mp4`, and `episode.npz`. Frame `i` in both videos matches row `i` in the joint state and action arrays. The archive also includes the task, frame rate, timestamps, reset seed, joint names, and units. These are raw demonstrations for later conversion to LeRobotDataset.

## Overhead camera placement

The camera's optical center is **48 cm forward and 58 cm above the SO-101 mounting origin**, with no sideways offset. It aims at a spot on the table **20 cm forward of the mount**, which puts it about **26° off vertical**. Its vertical field of view is **60°**. In the scene's coordinates, +X runs from the arm into the workspace and +Z points up. The mount origin sits at tabletop height. This camera stays fixed across resets.

## Use it in your code

```python
from cube_sim import make_sim

with make_sim(instruction="Put the red cube in the box.", workspace_camera="top") as env:
    obs, info = env.reset(seed=0)
    print(env.task_description)
    print(obs["pixels"]["top"].shape, obs["pixels"]["wrist"].shape)

    action = obs["agent_pos"].copy()
    action[0] += 10
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
```

Each `step` applies six absolute joint targets for 1/30 second of simulation. The order is shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, and gripper. The first five targets are in degrees and the gripper uses 0–100.

To plug in your policy, replace the example action with one predicted from `obs["pixels"]`, `obs["agent_pos"]`, and `env.task_description`. The demo uses scripted joint movement. The instruction is text for your policy, and the stock reward still counts all four cubes. For color-specific evaluation, score whether the requested cube is in the box and the others are outside.

## Control the simulator with ZED hand tracking

Use the same Python installation that already has the ZED Python API. On this workstation, that is Python 3.10. If you run from an editor, select that interpreter. If the hand model is missing, download it to the project folder.

```powershell
python -m pip install mediapipe
Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile .\hand_landmarker.task
python teleop_stereo.py
```

Close ZED Explorer before running the script. Hold your hand at a comfortable neutral position in view of the ZED and press **C**. Move your hand left/right, up/down, or toward/away from the camera to move the simulated gripper. Pinch your thumb and index finger to close it. Use **A/D** to rotate the wrist, **R** to reset the scene, and **Q/Esc** to quit. The gripper holds its last target if tracking or wrist depth is lost.
