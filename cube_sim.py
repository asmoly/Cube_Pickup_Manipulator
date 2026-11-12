"""Load Hugging Face's SO-101 tabletop scene as a single Gymnasium environment."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import mujoco
import numpy as np

REPO_ID = "lerobot/so101-tabletop-mujoco"
REVISION = "43a427cc783cd2c09445cf341bea6bb5bd9c22d1"
MODULE_NAME = "_so101_tabletop_mujoco"
OVERHEAD_POSITION = (0.48, 0.0, 0.58)
OVERHEAD_TARGET = (0.20, 0.0, 0.0)


"""
BASIC USAGE:

from cube_sim import make_sim

with make_sim(instruction="Put the red cube in the box.") as sim:
    obs, _ = sim.reset(seed=0)
    action = obs["agent_pos"].copy()
    action[0] += 10  # Turn the shoulder 10 degrees
    obs, reward, terminated, truncated, info = sim.step(action)
"""

def build_three_cube_model(build_model, scene, root):
    """Compile a copy of the scene with the red body removed."""
    xml = ElementTree.parse(scene)
    worldbody = xml.find("worldbody")
    worldbody.remove(worldbody.find("body[@name='red_cube']"))
    for texture in xml.findall("asset/texture[@file]"):
        texture.set("file", (scene.parent / texture.get("file")).resolve().as_posix())
    with TemporaryDirectory() as directory:
        scene_copy = Path(directory) / scene.name
        xml.write(scene_copy, encoding="utf-8")
        return build_model(scene_copy, root)


def make_sim(*, instruction=None, workspace_camera="front", image_size=(640, 480), include_red_cube=True):
    """Return an environment with reset(), step(), render(), and close().

    An instruction overrides the task text available to a policy. The stock
    simulator still scores how many of the four cubes are in the box.
    Set workspace_camera="top" for the tilted overhead view. image_size is (width, height).
    """
    if workspace_camera not in ("front", "top"):
        raise ValueError('workspace_camera must be "front" or "top"')

    module_name = MODULE_NAME if include_red_cube else MODULE_NAME + "_three_cubes"
    if module_name not in sys.modules:
        cache = Path(__file__).parent / ".sim-cache"
        scene = cache / f"models--{REPO_ID.replace('/', '--')}" / "snapshots" / REVISION
        required = ("env.py", "so101_tabletop.xml", "robotstudio_so101/so101.xml", "textures/wood.png")
        if not all((scene / filename).is_file() for filename in required):
            try:
                from huggingface_hub import snapshot_download
            except ModuleNotFoundError as exc:
                if exc.name != "huggingface_hub":
                    raise
                raise RuntimeError(
                    f"The SO-101 scene needs to be downloaded. Install huggingface_hub in {sys.executable}."
                ) from exc
            scene = Path(
                snapshot_download(
                    REPO_ID,
                    revision=REVISION,
                    cache_dir=cache,
                    allow_patterns=["env.py", "so101_tabletop.xml", "robotstudio_so101/*", "textures/*"],
                )
            )
        spec = spec_from_file_location(module_name, scene / "env.py")
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            if not include_red_cube:
                module.CUBE_BODIES = tuple(name for name in module.CUBE_BODIES if name != "red_cube")
                module.DEFAULTS["randomize"]["bodies"].pop("red_cube")
                build_model = module.build_model
                module.build_model = lambda scene, root: build_three_cube_model(build_model, scene, root)
        except Exception:
            del sys.modules[module_name]
            raise

    module = sys.modules[module_name]
    width, height = image_size
    options = {
        **module.DEFAULTS,
        "cameras": {workspace_camera: workspace_camera, "wrist": "wrist_cam"},
        "observation_width": width,
        "observation_height": height,
    }
    if workspace_camera == "front":
        return module.SO101Env(**options, instruction=instruction)

    class OverheadEnv(module.SO101Env):
        def _ensure_world(self):
            needs_camera_setup = self._world is None
            world = super()._ensure_world()
            if needs_camera_setup:
                camera = world.model.cam("top")
                camera.pos[:] = OVERHEAD_POSITION
                # The pinned scene restores positions from cam_pos0 on reset.
                world.cam_pos0[camera.id] = OVERHEAD_POSITION

                forward = np.subtract(OVERHEAD_TARGET, OVERHEAD_POSITION)
                forward /= np.linalg.norm(forward)
                right = np.cross(forward, (0, 0, 1))
                right /= np.linalg.norm(right)
                up = np.cross(right, forward)
                rotation = np.column_stack((right, up, -forward))
                mujoco.mju_mat2Quat(camera.quat, rotation.ravel())
                mujoco.mj_forward(world.model, world.data)
            return world

    return OverheadEnv(**options, instruction=instruction)
