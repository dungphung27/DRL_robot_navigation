from envs.obstacle import Obstacle

# Vật cản từ (100,100) đến (200,200)
obstacle = Obstacle(100, 100, 100, 100)

tests = [
    # Robot bên trái, tia hướng sang phải
    ((50, 150), 0, 50),

    # Robot phía trên, tia hướng xuống
    ((150, 50), -90, 50),

    # Robot phía dưới, tia hướng lên
    ((150, 250), 90, 50),

    # Robot bên phải, tia hướng sang trái
    ((250, 150), 180, 50),

    # Robot ở bên trong vật cản
    ((150, 150), 0, 0),
]

for pos, angle, expected in tests:
    distance = obstacle.calculate_intersection(
        angle,
        pos,
        500
    )

    print(
        f"Position={pos}, Angle={angle}, "
        f"Distance={distance:.2f}, Expected={expected}"
    )

    assert abs(distance - expected) < 1e-6

print("All sensor tests passed!")