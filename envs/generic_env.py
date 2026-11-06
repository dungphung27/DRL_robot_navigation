from tkinter import *
import numpy as np
from typing import List, Tuple
from .obstacle import Obstacle
from enum import IntEnum
import os
class RobotAction(IntEnum):
    FORWARD = 0, # vel = 1, rotation = 0
    ROTATE_LEFT = 1 # vel = 0, rotation = 3s
    ROTATE_RIGHT = 2 # vel = 0, rotation = -3
    

class GenericWorld: 
    

    def __init__(self, width: int, height: int, refresh_rate:int=10, manual:bool=False, see_all:bool=False) -> None:
        self.width = width
        self.height = height
        
        # ---- robot parameters
        self.pos = (width//2, height//2) # the position of the robot
        self.angle = 0 # the angle of the robot in degrees (0 represents the right/EAST direction)
        
        self.vel = 0 # the linear velocity
        self.rotational_vel = 0 # the angular velocity
        self.fov = 180 # the field of view of a robot (from its orientiation angle) (current 180 fov)
        
        # obstacles
        self._obstacles = []
        
        # the goal to goto
        self.goal = self._generate_random_position()
        self.goalbox = 50 # Easy mode: square goal region 50x50
        
        # the maximum view that the robot can see 
        # this assumes that the robot if possible can see the whole environment
        self.max_view = np.sqrt(self.width ** 2 + self.height ** 2) 
        
        # the distances objects are from the location of the robot
        # this is called the point cloud
        # the first index is the minimum most location of the point cloud
        self.point_cloud = np.ones(shape=self.fov//3+1) 
        
        # whethere to accept manual inputs
        self.manual = manual
        # whether to see all obstacles and not just point cloud
        self.see_all = see_all
        
        # refresh rate
        self.refresh_rate = refresh_rate
        self.tick = 0
        
        # rendering elmeents
        self.top = Tk()
        self.canvas = Canvas(self.top, height=height, width=width)
        
        # on key press
        if (manual):
            # background tick update
            self.top.after(self.refresh_rate, self._update)
            self.top.bind('<KeyPress>', self._onKeyPress)
    
        
    def _draw_cursor(self, size=20) -> None:
        coord = self.pos[0] - size//2, self.pos[1] - size//2, self.pos[0] + size, self.pos[1] + size
        self.canvas.create_arc(coord, start=157.5 + self.angle, extent=45, fill="red", tags=('robot'))


    def _draw_obstacles(self) -> None:
        for obstacle in self._obstacles:
            self.canvas.create_rectangle(obstacle.x1, obstacle.y1, obstacle.x2, obstacle.y2, fill='black', tags=('obstacle'))


    def _draw_goal(self) -> None: 
        box_side = self.goalbox / 2
        goal_rect = self.goal[0] - box_side, self.goal[1] - box_side, self.goal[0] + box_side, self.goal[1] + box_side
        self.canvas.create_rectangle(goal_rect, fill='green', tags=('goal'))      

    
    def _draw_fov(self, size=2) -> None:
        i = 0
        
        for fov_angle in range(self.angle - self.fov // 2, self.angle + self.fov // 2+1, 3):
            updated_pos = (self.pos[0] + np.cos(-fov_angle * np.pi/180) * self.point_cloud[i],
                           self.pos[1] + np.sin(-fov_angle * np.pi/180) * self.point_cloud[i])
            self.canvas.create_oval(updated_pos[0] - size/2, updated_pos[1] - size/2, 
                                    updated_pos[0] + size/2, updated_pos[1] + size/2, fill='black', tags='fov')
            i += 1


    def _update_position(self) -> None:
        # update the angle first 
        self.angle = (self.angle + self.rotational_vel) % 360
        
        updated_vel = (self.vel * np.cos(-self.angle * np.pi / 180), 
                       self.vel * np.sin(-self.angle * np.pi / 180))
        updated_pos = (self.pos[0] + updated_vel[0], self.pos[1] + updated_vel[1])        
        
        if (self.manual):
            if not self.has_collision(updated_pos):
                self.pos = updated_pos
        else:
            self.pos = updated_pos
            
    
    def _min_distance_to_wall(self, fov_angle):
        """
        Tính khoảng cách từ robot đến tường
        theo hướng của một tia cảm biến.

        Hệ tọa độ:
            X dương: sang phải
            Y dương: xuống dưới
            Góc 0°: sang phải
            Góc 90°: lên trên
        """

        x, y = self.pos

        # Chuyển độ sang radian
        angle_rad = np.deg2rad(fov_angle)

        # Vector hướng đơn vị
        dx = np.cos(angle_rad)
        dy = -np.sin(angle_rad)

        distances = []

        eps = 1e-12

        # 1. Giao với tường trái và phải
        if abs(dx) > eps:

            # Tường trái: x = 0
            t_left = (0 - x) / dx

            if t_left >= 0:
                y_hit = y + t_left * dy

                if 0 <= y_hit <= self.height:
                    distances.append(t_left)

            # Tường phải: x = width
            t_right = (self.width - x) / dx

            if t_right >= 0:
                y_hit = y + t_right * dy

                if 0 <= y_hit <= self.height:
                    distances.append(t_right)

        # 2. Giao với tường trên và dưới
        if abs(dy) > eps:

            # Tường trên: y = 0
            t_top = (0 - y) / dy

            if t_top >= 0:
                x_hit = x + t_top * dx

                if 0 <= x_hit <= self.width:
                    distances.append(t_top)

            # Tường dưới: y = height
            t_bottom = (self.height - y) / dy

            if t_bottom >= 0:
                x_hit = x + t_bottom * dx

                if 0 <= x_hit <= self.width:
                    distances.append(t_bottom)

        # 3. Chọn giao điểm gần nhất
        if distances:
            return min(min(distances), self.max_view)

        return self.max_view
    
    def _calculate_distance_from(self, fov_angle: int, index: int) -> float:        
        constants = (self.pos[0], self.pos[1])
        
        min_distance = self._min_distance_to_wall(fov_angle)
        for obstacle in self._obstacles:
            test_distance = obstacle.calculate_intersection(fov_angle, constants, self.max_view)

            if (test_distance < min_distance):
                min_distance = test_distance
            
        self.point_cloud[index] = min_distance
        return min_distance
    
    
    def _update_point_cloud(self) -> None:
        fov_angles = range(
            self.angle - self.fov // 2,
            self.angle + self.fov // 2 + 1,
            3
        )

        for i, fov_angle in enumerate(fov_angles):
            self._calculate_distance_from(fov_angle, i)

    def _update_data(self) -> None:
        self._update_position()
        self._update_point_cloud()


    def _update(self) -> None:
        # print(self.angle, self.point_cloud[0], self.point_cloud[90], self.point_cloud[180])
        # print(self.point_cloud)
        # print(self.pos, self.vel, self.angle)
        # reset canvas
        self.canvas.delete("all")
        
        # update all data
        self._update_data()
        
        # re-render pos and obstacles
        self._draw_cursor()
        self._draw_goal()
        if (self.see_all):
            self._draw_obstacles()
        self._draw_fov()

        self.canvas.update()
        self.canvas.pack()
        if (self.manual):
            self.top.after(self.refresh_rate, self._update)
        
        
    def _onKeyPress(self, event) -> None:
        if (event.keysym == "Left"):
            print('rotating')
            self.vel = 0
            self.rotational_vel = 1
        elif (event.keysym == "Right"):
            print('rotating')
            self.vel = 0 
            self.rotational_vel = -1
        elif (event.keysym == "Up"):
            print('moving forward')
            self.rotational_vel = 0
            self.vel = 1
        elif (event.keysym == "Down"):
            print('moving backwards')
            self.rotational_vel = 0
            self.vel = -1
        elif (event.keysym == '0'):
            print('0 pressed | velocity reset | angle reset')
            self.vel = 0
            self.rotational_vel = 0


    def render(self) -> None:
        self.top.mainloop()
    
    
    def add_obstacles(self, obstacles: List[Obstacle]) -> None:
        for obstacle in obstacles:
            self._obstacles.append(obstacle)    


    def has_collision(self, position) -> bool:
        collides = any([obstacle.collides_with(position) for obstacle in self._obstacles])
        
        on_border = False
        if (position[0] <= 0 or position[0] >= self.width or position[1] <= 0 or position[1] >= self.height):
            on_border = True
        
        return collides or on_border


    def _generate_random_position(self) -> Tuple[int, int]:
        """Choose a robot start away from walls and obstacles."""
        margin = min(10, max(1, min(self.width, self.height) // 10))
        for _ in range(2000):
            x = int(np.random.randint(margin, self.width - margin))
            y = int(np.random.randint(margin, self.height - margin))
            if not self.has_collision((x, y)):
                return (x, y)
        raise RuntimeError("Cannot find collision-free robot start")


    def _within_goal(self) -> bool:
        box_side = self.goalbox / 2
        if ((self.pos[0] >= self.goal[0] - box_side and self.pos[0] <= self.goal[0] + box_side)
            and (self.pos[1] >= self.goal[1] - box_side and self.pos[1] <= self.goal[1] + box_side)):
            return True
        else:
            return False
    
                
    def _generate_goal_position(self) -> Tuple[int, int]:
        """Choose a goal square inside the map, clear of obstacles and robot."""
        half = self.goalbox / 2
        low_x, high_x = int(np.ceil(half)), int(np.floor(self.width - half))
        low_y, high_y = int(np.ceil(half)), int(np.floor(self.height - half))
        if low_x > high_x or low_y > high_y:
            raise ValueError("Goal area does not fit in the environment")

        for _ in range(2000):
            gx = int(np.random.randint(low_x, high_x + 1))
            gy = int(np.random.randint(low_y, high_y + 1))
            if np.hypot(gx - self.pos[0], gy - self.pos[1]) < max(70, self.goalbox * 1.5):
                continue
            # Keep the entire 50x50 goal region away from all obstacles.
            overlaps = any(
                gx + half >= o.x1 and gx - half <= o.x2
                and gy + half >= o.y1 and gy - half <= o.y2
                for o in self._obstacles
            )
            if not overlaps:
                return (gx, gy)
        raise RuntimeError("Cannot place a free goal; check map or goal size")

    def _goal_distance(self) -> float:
        return float(np.hypot(self.goal[0] - self.pos[0],
                              self.goal[1] - self.pos[1]))

    def _relative_goal_angle(self) -> float:
        # Screen coordinates: y grows downward, while heading 90 degrees points up.
        dx = self.goal[0] - self.pos[0]
        dy = self.goal[1] - self.pos[1]
        goal_heading = np.degrees(np.arctan2(-dy, dx))
        return float((goal_heading - self.angle + 180) % 360 - 180)

    def _observation(self) -> np.ndarray:

        # Giữ nguyên state cũ theo mặc định.
        encoding = os.getenv("DQN_STATE_ENCODING", "sincos").lower()

        if encoding == "sincos":
            relative_angle = np.deg2rad(
                self._relative_goal_angle()
            )

            goal_features = np.array([
                self._goal_distance() / self.max_view,
                np.sin(relative_angle),
                np.cos(relative_angle)
            ], dtype=np.float32)

            lidar_features = (
                np.asarray(self.point_cloud, dtype=np.float32)
                / self.max_view
            )

            return np.concatenate([
                goal_features,
                lidar_features
            ]).astype(np.float32)

        # State cũ: dùng cho các checkpoint đã huấn luyện.
        return np.concatenate((
            np.array([
                self._goal_distance(),
                self.angle,
                self._relative_goal_angle()
            ], dtype=np.float32),

            np.asarray(self.point_cloud, dtype=np.float32)
        ))

    def reset(self, render: bool = False) -> np.ndarray:
        """Start a new episode without taking an extra movement step."""
        self.pos = self._generate_random_position()
        self.angle = 0
        self.vel = 0
        self.rotational_vel = 0
        self.goal = self._generate_goal_position()
        self._update_point_cloud()

        if render or self.manual:
            # Draw without running _update(), which also advances movement.
            self.canvas.delete("all")
            self._draw_cursor()
            self._draw_goal()
            if self.see_all:
                self._draw_obstacles()
            self._draw_fov()
            self.canvas.pack()
            self.canvas.update()

        return self._observation()


    def step(self, action: RobotAction, render: bool = False):
        # Measure progress BEFORE applying action.
        previous_distance = self._goal_distance()

        if action == RobotAction.FORWARD:
            self.vel, self.rotational_vel = 1, 0
        elif action == RobotAction.ROTATE_LEFT:
            self.vel, self.rotational_vel = 0, 1
        elif action == RobotAction.ROTATE_RIGHT:
            self.vel, self.rotational_vel = 0, -1
        else:
            raise ValueError(f"Invalid action: {action}")
        # Khoảng cách đến đích trước khi thực hiện hành động
        if render or self.manual:
            self._update()
        else:
            self._update_data()
        # Khoảng cách đến đích sau khi thực hiện hành động
        new_distance = self._goal_distance()
        # Termination priority: collision first, then goal.
        collision = self.has_collision(self.pos)
        reached_goal = self._within_goal()
        done = bool(collision or reached_goal)

        if reached_goal:
            reward = 100.0
            done = True

        elif collision:
            reward = -100.0
            done = True

        else:
            reward = 2.0 * (previous_distance - new_distance) - 0.1
            done = False

        return self._observation(), float(reward), done


if __name__ == "__main__":
    world = GenericWorld(500, 500, manual=True)
    world.render()
    # world.reset()
    
    # print(world.step(RobotAction.FORWARD))
    # print(world.step(RobotAction.FORWARD))
    # print(world.step(RobotAction.FORWARD))
    # print(world.step(RobotAction.FORWARD))
    # print(world.step(RobotAction.FORWARD))
    # print(world.step(RobotAction.FORWARD))
    # print(world.step(RobotAction.FORWARD))



