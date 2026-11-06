from typing import Tuple
import numpy as np

class Obstacle: 
    
    
    def __init__(self, x, y, width, height) -> None:
        self.x1 = x
        self.y1 = y
        self.x2 = x + width
        self.y2 = y + height
        self.intersections = []
        
    def collides_with(self, position: Tuple[int, int]) -> bool:
        if ((position[0] >= self.x1 and position[0] <= self.x2) and 
            (position[1] >= self.y1 and position[1] <= self.y2)):
            return True
        else:
            return False
    
    
    def dist_interval(self, current_pos: Tuple[float, float]) -> Tuple[float, float]:
        test_points = [(self.x1, self.y1), (self.x1, self.y2), (self.x2, self.y1), (self.x2, self.y2)]
        distance_betweens = [self.dist_between(current_pos, test) for test in test_points]
        min_dist = min(distance_betweens)
        max_dist = max(distance_betweens)
        return min_dist, max_dist
    
    
    def find_intersects(self, intersection) -> float:
        update_pos = (self.current_pos[0] + intersection * np.cos(-self.fov_angle * np.pi/180), 
                        self.current_pos[1] + intersection * np.sin(-self.fov_angle * np.pi/180))
        if (self.collides_with(update_pos)):
            self.intersections.append(intersection)
            return intersection
        return 0
        
    def calculate_intersection(self, fov_angle, current_pos, max_view):
        """
        Tính khoảng cách từ robot tới hình chữ nhật
        bằng thuật toán Ray-AABB Intersection.
        """

        x0, y0 = current_pos

        # Chuyển góc sang radian
        angle_rad = np.deg2rad(fov_angle)

        # Vector hướng của tia
        dx = np.cos(angle_rad)
        dy = -np.sin(angle_rad)

        t_min = 0.0
        t_max = float(max_view)

        # Kiểm tra giao điểm theo từng trục
        for origin, direction, lower, upper in (
            (x0, dx, self.x1, self.x2),
            (y0, dy, self.y1, self.y2)
        ):

            # Tia gần như song song với trục đang xét
            if abs(direction) < 1e-12:
                if origin < lower or origin > upper:
                    return max_view
                continue

            t1 = (lower - origin) / direction
            t2 = (upper - origin) / direction

            t_near = min(t1, t2)
            t_far = max(t1, t2)

            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)

            if t_min > t_max:
                return max_view

        return min(t_min, max_view)

    @staticmethod
    def absolute_min(i1, i2):
        if (abs(i1) < abs(i2)):
            return i1
        else:
            return i2
        
    @staticmethod
    def dist_between(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return ((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2) ** 0.5
    
    
if __name__ == "__main__":
    o = Obstacle(100, 100, 50, 50)
    print(o.calculate_intersection(-90, (20, 20), 500 * np.sqrt(2)))