from .generic_env import GenericWorld
from .obstacle import Obstacle


class ObstaclesWorld(GenericWorld):
    """Easy map: two small rectangular obstacles."""

    def __init__(self, width: int, height: int, refresh_rate: int = 10,
                 manual: bool = False, see_all: bool = False) -> None:
        super().__init__(width, height, refresh_rate, manual, see_all)

        self.add_obstacles([
            Obstacle(190, 180, 20, 20),
        ])
        # Reset only after obstacle placement so start/goal are valid.
        self.reset()


if __name__ == "__main__":
    world = ObstaclesWorld(300, 300, manual=True, see_all=True)
    world.render()
