"""Drive the simulated SO-101 with an Xbox controller on Windows."""

import ctypes
from time import perf_counter

import cv2
import mujoco
import numpy as np

from cube_sim import make_sim

WINDOW = "SO-101 | Xbox controller"
MOVE_SPEED = 0.12  # meters per second
ROLL_SPEED = np.deg2rad(90)  # radians per second speed of gripper rotation
READY_POSE = (0, -50, 40, 85, -90, 65) # Starting joint angles

# Sim Notes:
# Simulator +X points forward from the base, +Y points left, and +Z points up.
# Controller input:
#   - Left stick controls movement along xy plane
#   - Right stick up/down controller height (z)
#   - Right stick left_right controller gripper rotation
#   - Right trigger closes and opens the gripper

class Gamepad(ctypes.Structure):
    _fields_ = [
        ("buttons", ctypes.c_uint16),
        ("left_trigger", ctypes.c_uint8),
        ("right_trigger", ctypes.c_uint8),
        ("left_x", ctypes.c_int16),
        ("left_y", ctypes.c_int16),
        ("right_x", ctypes.c_int16),
        ("right_y", ctypes.c_int16),
    ]

class ControllerState(ctypes.Structure):
    _fields_ = [("packet", ctypes.c_uint32), ("gamepad", Gamepad)]

def read_controller(get_state):
    """Returns the current state of the controller"""
    state = ControllerState()
    for index in range(4): # Loops through the 4 available controler ports to find the connected one
        if get_state(index, ctypes.byref(state)) == 0:
            return state.gamepad
        
    return None

def stick(value):
    """Scale the stick to -1..1, ignoring small movements near the center."""
    return np.sign(value)*max(0, min(abs(value), 32767) - 8000)/(32767 - 8000)

def move_target(gamepad, target, roll, dt):
    """Returns a new target xyz position for the gripper based on controller inputs"""
    direction = np.array([stick(gamepad.left_y), -stick(gamepad.left_x), stick(gamepad.right_y)]) # Gets the direction vector based on controller input
    target = np.clip(target + direction*MOVE_SPEED*dt, (0.10, -0.20, 0.018), (0.34, 0.20, 0.23)) # This keeps the target position inside of the designated box
    roll += stick(gamepad.right_x)*ROLL_SPEED*dt # Sets new target roll
    gripper = 65*(1 - gamepad.right_trigger/255) # Sets gripper state 
    return target, roll, gripper


class GripperIK:
    """Convert a gripper position and roll into six simulator joint targets."""
    def __init__(self, env):
        self.world = env._world
        self.model = self.world.model
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("gripperframe").id
        self.positions = self.world.qadr[:5]
        names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
        self.dofs = [self.model.joint(name).dofadr[0] for name in names]
        self.jacobian = np.zeros((3, self.model.nv))
        self.damping = 0.015**2 * np.eye(4)

    def tip(self):
        """Gets grippers xyz position in meters"""
        return self.world.data.site_xpos[self.site].copy()

    def command(self, target, previous, roll, gripper):
        """
        Moves arm towards target pos by changing joint angles, tracks previous angles to minimize rapid movements
        previous - the previous joint angles of the arm
        """
        angles = np.deg2rad(previous[:5]).astype(np.float64)
        angles[4] = roll # Sets new roll angle
        low, high = self.world.lo[:4], self.world.hi[:4] # These are the min and max angles for each joint
        pitch_weight = 0.06 # This is how strongly the solver tries to hold the arm at 75 degree angle

        # Inverse kinematics loop
        for _ in range(6):
            self.data.qpos[self.positions] = angles
            mujoco.mj_fwdPosition(self.model, self.data)
            # Keep the jaws tilted toward the tabletop while moving the tip.
            pitch_error = np.deg2rad(75) - angles[1:4].sum()
            position_error = target - self.data.site_xpos[self.site]
            mujoco.mj_jacSite(self.model, self.data, self.jacobian, None, self.site)
            jacobian = np.vstack((self.jacobian[:, self.dofs], (0, pitch_weight, pitch_weight, pitch_weight)))
            error = np.append(position_error, pitch_error * pitch_weight)
            update = np.linalg.solve(jacobian.T @ jacobian + self.damping, jacobian.T @ error)
            angles[:4] = np.clip(angles[:4] + np.clip(update, -0.12, 0.12), low, high)

        action = previous.copy()
        action[:5] += np.clip(np.rad2deg(angles) - previous[:5], -4, 4)
        action[5] = gripper
        return action


def run_demo(env, get_state):
    env.initial_joints[:] = READY_POSE # Sets initial angles
    obs, _ = env.reset(seed=0) # Sets up sim
    ik = GripperIK(env) # Creates the inverse kinematics object

    # Gets current action (current joint angles) and x,y,z pos, as well as roll
    action = obs["agent_pos"]
    target = ik.tip()
    roll = np.deg2rad(action[4])

    dt = 1/env.fps
    finished = False

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280*2, 480*2)

    while True:
        start = perf_counter()
        gamepad = read_controller(get_state)

        status = "Running"
        if finished:
            status = "Episode ended | Press R to reset"
        elif gamepad is None:
            status = "Connect an Xbox controller | Q: quit"
        else:
            target, roll, gripper = move_target(gamepad, target, roll, dt) # Get new xyz pos
            roll = np.clip(roll, ik.world.lo[4], ik.world.hi[4]) # Clip the roll to be within the min and max angles
            action = ik.command(target, action, roll, gripper) # Run inverse kinematics to get new joint angles
            obs, _, terminated, truncated, _ = env.step(action) # Take a step in the environment
            finished = terminated or truncated

        # Rendering
        pixels = obs["pixels"]
        preview = cv2.cvtColor(np.hstack((pixels["top"], pixels["wrist"])), cv2.COLOR_RGB2BGR)
        cv2.putText(preview, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)

        cv2.imshow(WINDOW, preview)
        delay_ms = max(1, round(1000 * (dt - (perf_counter() - start))))
        key = cv2.waitKey(delay_ms) & 0xFF

        # Window closing
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("r"): # Reset check
            obs, _ = env.reset()
            action = obs["agent_pos"]
            target = ik.tip()
            roll = np.deg2rad(action[4])
            finished = False


def main():
    get_state = ctypes.WinDLL("xinput1_4").XInputGetState
    get_state.argtypes = (ctypes.c_uint32, ctypes.POINTER(ControllerState))
    get_state.restype = ctypes.c_uint32

    try:
        with make_sim(workspace_camera="top", image_size=(320*2, 240*2)) as env:
            run_demo(env, get_state)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
