# launch.py
import os, sys, time, argparse
import gymnasium as gym
import numpy as np
import torch

# 1) Import your env module BEFORE gym.make so @register_env runs
#    (adjust import path to where multiple_tasks_env.py lives)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.mani_skill.tasks import multiple_tasks_env  # noqa: F401

def make_env(
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    robot_uids="so100",
    render_mode="human",
):
    env = gym.make(
        "Combined-v1",
        obs_mode=obs_mode,
        control_mode=control_mode,
        robot_uids=robot_uids,
        render_mode=render_mode,     # <-- this is key
        # num_envs=1,                # keep single env for GUI preview
    )
    return env

def preview_gui():
    # If you previously ran offscreen/EGL, undo it for GUI:
    os.environ.pop("PYOPENGL_PLATFORM", None)

    env = make_env(render_mode="human")
    print(env.action_space, env.observation_space)
    obs, info = env.reset(seed=0)
    print("Robot:", env.unwrapped.robot_uids)
    ua = env.unwrapped
    print("CubeA:", ua.CubeA.pose.p, "CubeB:", ua.CubeB.pose.p)

    # Zero action (correct shape) so the scene stays still
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    try:
        for t in range(600):  # ~10 seconds at 60 FPS
            obs, rew, term, trunc, info = env.step(action)
            env.render()  # draw a frame to the SAPIEN window
            if t % 60 == 0:
                print(f"[{t}] push_goal:", ua.push_goal_region.pose.p)
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

def record_mp4(out_path="combined_preview.mp4", steps=600, fps=60):
    # Offscreen render for headless servers
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    env = make_env(render_mode="rgb_array")
    obs, info = env.reset(seed=0)

    frames = []
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    for t in range(steps):
        obs, rew, term, trunc, info = env.step(action)
        frame = env.render()  # returns (H, W, 3) uint8
        if frame is not None:
            frames.append(frame)

    env.close()

    if frames:
        import imageio.v2 as imageio
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"Saved video to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gui", "video"], default="gui")
    parser.add_argument("--out", default="combined_preview.mp4")
    args = parser.parse_args()

    if args.mode == "gui":
        preview_gui()
    else:
        record_mp4(out_path=args.out)

if __name__ == "__main__":
    main()
