from envs.obstacles_env import ObstaclesWorld

env = ObstaclesWorld(500, 500, see_all=True)

env.reset(render=False)
env.pos = (250, 250)

angles = [0, 90, 180, 270, 360, 450]

for angle in angles:
    distance = env._min_distance_to_wall(angle)

    print(
        f"Angle: {angle:3d}° | "
        f"Distance: {distance}"
    )