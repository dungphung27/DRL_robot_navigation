from envs.obstacles_env import ObstaclesWorld
from envs.generic_env import RobotAction

env = ObstaclesWorld(500, 500, see_all=True)

# Kiểm tra reset
state = env.reset(render=False)

print("State shape:", state.shape)
assert state.shape == (63,)

# Kiểm tra step
next_state, reward, done = env.step(
    RobotAction.FORWARD,
    render=False
)

print("Next state shape:", next_state.shape)
print("Reward:", reward)
print("Done:", done)

assert next_state.shape == (63,)

# Kiểm tra va chạm và việc không tự reset
env.reset(render=False)

env.pos = (1.0, 250.0)
env.angle = 180

# Đặt mục tiêu ở xa vị trí kiểm tra
env.goal = (450, 450)

old_goal = env.goal

next_state, reward, done = env.step(
    RobotAction.FORWARD,
    render=False
)

print("Collision test:")
print("Position:", env.pos)
print("Reward:", reward)
print("Done:", done)

assert done is True
assert reward == -100.0
assert env.goal == old_goal
assert env.pos[0] <= 0

print("All episode tests passed!")