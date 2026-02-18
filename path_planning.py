import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Polygon
import heapq
import random

#simulating everything in inches

class Environment:

    def __init__(self, width=72, height=54):
        self.width = width
        self.height = height
        self.obstacles = []
        self.start_zone = (0, 0, 10, 10)
        self.goal_zone = (62, 44, 10, 10)
        
    def generate_obstacles(self, num_obstacles=5, obstacle_size=3):
        self.obstacles = []
        
        while len(self.obstacles) < num_obstacles:
            x = random.uniform(0, self.width-obstacle_size)
            y = random.uniform(0, self.height-obstacle_size)
            
            if self.zone_intersect(x, y, obstacle_size, self.start_zone):
                continue
            
            if self.zone_intersect(x, y, obstacle_size, self.goal_zone):
                continue
            
            self.obstacles.append((x, y, obstacle_size, obstacle_size))
    
    def zone_intersect(self, x, y, size, zone):
        zone_x, zone_y, zone_w, zone_h = zone
        if (x + size < zone_x or x > zone_x + zone_w or 
            y + size < zone_y or y > zone_y + zone_h):
            return False
        return True


class ThreeLinkRobot:
    def __init__(self, x, y, alpha1, alpha2, link_width=2.5, link_length=5): #arbitraily chose 2.5 and 5 too make not too big
        self.x = x
        self.y = y
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.link_width = link_width
        self.link_length = link_length
        
    def get_state(self):
        return (self.x, self.y, self.alpha1, self.alpha2)
    
    def set_state(self, state):
        self.x, self.y, self.alpha1, self.alpha2 = state
    
    def get_link_positions(self):
        links = []
        
        link1_corners = self.get_link_corners(self.x, self.y, self.alpha1)
        links.append(link1_corners)
        
        end1_x = self.x+self.link_length*np.cos(np.radians(self.alpha1))
        end1_y = self.y+self.link_length*np.sin(np.radians(self.alpha1))
        
        total_angle2 = self.alpha1 + self.alpha2
        link2_corners = self.get_link_corners(end1_x, end1_y, total_angle2)
        links.append(link2_corners)
        
        end2_x = end1_x+self.link_length*np.cos(np.radians(total_angle2))
        end2_y = end1_y+self.link_length*np.sin(np.radians(total_angle2))
        
        link3_corners = self.get_link_corners(end2_x, end2_y, total_angle2)
        links.append(link3_corners)
        
        return links
    
    def get_link_corners(self, x, y, angle):
        half_width = self.link_width/2
        half_length = self.link_length/2
        
        corners = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])
        
        theta = np.radians(angle)
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        rotated = corners@rotation.T
    
        offset_x = half_length*np.cos(theta)
        offset_y = half_length*np.sin(theta)
        
        translated = rotated+np.array([x+offset_x, y+offset_y])
        
        return translated



class PathPlanner:
    def __init__(self, environment, robot):
        self.env = environment
        self.robot = robot
        self.position_step = 2.0
        self.angle_step = 30
        
    def check_collision(self, state):
        self.robot.set_state(state)
        links = self.robot.get_link_positions()
        
        for link in links:
            for corner in link:
                if (corner[0] < 0 or corner[0] > self.env.width or
                    corner[1] < 0 or corner[1] > self.env.height):
                    return True
        
        for obstacle in self.env.obstacles:
            ox, oy, ow, oh = obstacle
            obstacle_box = (ox, oy, ox + ow, oy + oh)
            
            for link in links:
                link_box = self.get_box(link)
                
                if self.rectangles_intersect(link_box, obstacle_box):
                    return True
        
        return False
    
    def get_box(self, shape):

        min_x = np.min(shape[:, 0])
        min_y = np.min(shape[:, 1])
        max_x = np.max(shape[:, 0])
        max_y = np.max(shape[:, 1])
        
        return (min_x, min_y, max_x, max_y)
    
    def rectangles_intersect(self, rect1, rect2):
        min1_x, min1_y, max1_x, max1_y = rect1
        min2_x, min2_y, max2_x, max2_y = rect2
        
        if (max1_x < min2_x or
            min1_x > max2_x or
            max1_y < min2_y or
            min1_y > max2_y):
            return False
        
        return True
    
    
    def heuristic(self, state, goal_state):
        x1, y1, _, _ = state
        x2, y2, _, _ = goal_state
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    def get_neighbors(self, state):
        x, y, alpha1, alpha2 = state
        neighbors = []
        
        #8 point connectivity
        for dx, dy in [(self.position_step, 0), 
                       (-self.position_step, 0), 
                       (0, self.position_step), 
                       (0, -self.position_step),
                       (self.position_step, self.position_step), 
                       (-self.position_step, -self.position_step),
                       (self.position_step, -self.position_step), 
                       (-self.position_step, self.position_step)]: 
            neighbors.append((x + dx, y + dy, alpha1, alpha2))
        
        #8 point angle connectivity
        for dalpha1 in [-self.angle_step, 0, self.angle_step]:
            for dalpha2 in [-self.angle_step, 0, self.angle_step]:
                if dalpha1 == 0 and dalpha2 == 0:
                    continue
                new_alpha1 = (alpha1 + dalpha1) % 360
                new_alpha2 = (alpha2 + dalpha2) % 360
                neighbors.append((x, y, new_alpha1, new_alpha2))
        
        return neighbors
    
    def collapse_state(self, state):
        x, y, alpha1, alpha2 = state
        new_x = round(x/self.position_step)*self.position_step
        new_y = round(y/self.position_step)*self.position_step
        new_alpha1 = round(alpha1/self.angle_step)*self.angle_step % 360
        new_alpha2 = round(alpha2/self.angle_step)*self.angle_step % 360
        return (new_x, new_y, new_alpha1, new_alpha2)
    
    def a_star_search(self, start_state, goal_state):
        start_state = self.collapse_state(start_state)
        goal_state = self.collapse_state(goal_state)
        
        counter = 0
        open_set = [(0, counter, start_state)]
        counter += 1
        
        came_from = {}
        g_score = {start_state: 0}
        f_score = {start_state: self.heuristic(start_state, goal_state)}
        
        closed_set = set()
        
        iterations = 0
        max_iterations = 10000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            if iterations % 100 == 0:
                print(f"Iteration {iterations}, Open set size: {len(open_set)}")
            
            _, _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue

            x, y, _, _ = current
            gx, gy, _, _ = goal_state
            if abs(x - gx) <= self.position_step and abs(y - gy) <= self.position_step:
                print(f"Goal reached in {iterations} iterations!")
                return self.reconstruct_path(came_from, current)
            
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current):
                neighbor = self.collapse_state(neighbor)
                
                if neighbor in closed_set:
                    continue
                
                if self.check_collision(neighbor):
                    continue
                
                new_g = g_score[current] + 1
                
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = new_g
                    f = new_g + self.heuristic(neighbor, goal_state)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1
        
        print(f"No path found after {iterations} iterations")
        return None
    
    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


def visualize_path(env, robot, path):
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    start_rect = Rectangle((env.start_zone[0], env.start_zone[1]), 
                           env.start_zone[2], env.start_zone[3],
                           linewidth=2, edgecolor='green', facecolor='lightgreen', 
                           alpha=0.3, label='Start Zone')
    ax.add_patch(start_rect)
    
    goal_rect = Rectangle((env.goal_zone[0], env.goal_zone[1]), 
                          env.goal_zone[2], env.goal_zone[3],
                          linewidth=2, edgecolor='red', facecolor='lightcoral', 
                          alpha=0.3, label='Goal Zone')
    ax.add_patch(goal_rect)
    
    for obstacle in env.obstacles:
        ox, oy, ow, oh = obstacle
        obs_rect = Rectangle((ox, oy), ow, oh, 
                            linewidth=1, facecolor='black', alpha=0.7)
        ax.add_patch(obs_rect)
    
    if path:
        path_x = [state[0] for state in path]
        path_y = [state[1] for state in path]
        ax.plot(path_x, path_y, 'b--', linewidth=2, alpha=0.5, label='Path')

        num_positions = len(path)
        indices = np.linspace(0, len(path)-1, num_positions, dtype=int)
        
        for i in indices:
            robot.set_state(path[i])
            links = robot.get_link_positions()
            
            for link in links:
                poly = Polygon(link, alpha=0.3, facecolor='lightblue', 
                               edgecolor='blue', linewidth=1)
                ax.add_patch(poly)
        
        robot.set_state(path[-1])
        links = robot.get_link_positions()
        for link in links:
            poly = Polygon(link, alpha=0.8, facecolor='red', 
                           edgecolor='lightcoral', linewidth=2)
            ax.add_patch(poly)
    
    ax.set_xlabel('X (inches)')
    ax.set_ylabel('Y (inches)')
    ax.set_title('3-Link Robot Path Planning with RRTs')
    ax.legend()
    
    plt.tight_layout()
    plt.show()



def print_path_steps(path):
    print(f"Total number of steps: {len(path)}\n")
    
    for i, state in enumerate(path):
        x, y, alpha1, alpha2 = state
        print(f"Step {i:2d}: Position=({x:5.2f}, {y:5.2f}), "
              f"Alpha1={alpha1:6.1f}°, Alpha2={alpha2:6.1f}°")
    


def main():
    #random.seed(42)
    #np.random.seed(42)
    
    env = Environment(width=72, height=54)
    env.generate_obstacles(num_obstacles=5, obstacle_size=3)
    
    print(f"Environment: {env.width} x {env.height} inches")
    print(f"Number of obstacles: {len(env.obstacles)}")
    
    start_x = random.uniform(2, 8)
    start_y = random.uniform(2, 8)
    start_alpha1 = random.uniform(0, 360)
    start_alpha2 = random.uniform(0, 360)
    
    print(f"\nStarting position: ({start_x:.2f}, {start_y:.2f})")
    print(f"Starting angles: Alpha1={start_alpha1:.1f}°, Alpha2={start_alpha2:.1f}°")
    
    robot = ThreeLinkRobot(start_x, start_y, start_alpha1, start_alpha2)
    
    goal_x = 65
    goal_y = 47
    goal_alpha1 = 0
    goal_alpha2 = 0
    
    planner = PathPlanner(env, robot)
    
    start_state = robot.get_state()
    goal_state = (goal_x, goal_y, goal_alpha1, goal_alpha2)
    
    if planner.check_collision(start_state):
        print("\nStart state is in collision")
    else:
        print("\nStart state is collision-free")
    
    if planner.check_collision(goal_state):
        print("Goal state is in collision")
    else:
        print("Goal state is collision-free")
    
    path = planner.a_star_search(start_state, goal_state)
    
    if path:
        print(f"\nPath found with {len(path)} steps")
        print_path_steps(path)
        visualize_path(env, robot, path)
    else:
        print("\nNo path found")
        visualize_path(env, robot, None)


if __name__ == "__main__":
    main()