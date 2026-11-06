"""Run from the repository root after copying the two env files."""
import numpy as np
from envs.obstacles_env import ObstaclesWorld
from envs.generic_env import RobotAction

env = ObstaclesWorld(300, 300, see_all=True)
try:
    for _ in range(30):
        state = env.reset(render=False)
        assert state.shape == (64,), state.shape
        assert np.isfinite(state).all()
        assert not env.has_collision(env.pos)
        assert not env._within_goal()
        previous_pos = env.pos
        next_state, reward, done = env.step(RobotAction.FORWARD, render=False)
        assert next_state.shape == (64,), next_state.shape
        assert np.isfinite(next_state).all()
        assert np.isfinite(reward)
        # Goal bearing should be between -180 and 180.
        assert -180 <= next_state[2] <= 180

    # Geometry test: robot points right, goal right -> bearing 0.
    env.pos = (100., 150.)
    env.goal = (200., 150.)
    env.angle = 0
    assert abs(env._relative_goal_angle()) < 1e-6
    env.goal = (0., 150.)
    assert abs(env._relative_goal_angle() + 180) < 1e-6
    print("PASS: state (64,), reward, random resets, goal bearing")
finally:
    env.top.destroy()
