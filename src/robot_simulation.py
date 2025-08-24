import numpy as np
import matplotlib.pyplot as plt

class DifferentialDriveRobot:
    """A class to simulate a differential drive robot."""
    
    def __init__(self, x=0, y=0, theta=0, wheel_base=20):
        """
        Initialize the robot's state.
        Args:
            x (float): Initial x-coordinate (pixels)
            y (float): Initial y-coordinate (pixels)
            theta (float): Initial orientation (radians)
            wheel_base (float): Distance between wheels (pixels) - affects turning
        """
        self.x = x
        self.y = y
        self.theta = theta  # Orientation in radians
        self.wheel_base = wheel_base
        self.path_x = [x]  # History of x positions
        self.path_y = [y]  # History of y positions
        self.path_theta = [theta] # History of orientations
        
    def set_pose(self, x, y, theta):
        """Set the robot's current pose and update history."""
        self.x = x
        self.y = y
        self.theta = theta
        self.path_x.append(x)
        self.path_y.append(y)
        self.path_theta.append(theta)
        
    def move(self, v_left, v_right, dt=1.0):
        """
        Move the robot based on kinematic equations for differential drive.
        Args:
            v_left (float): Left wheel velocity (pixels per second)
            v_right (float): Right wheel velocity (pixels per second)
            dt (float): Time step for movement (seconds)
        """
        # Calculate linear and angular velocity
        v = (v_right + v_left) / 2  # Linear velocity
        omega = (v_right - v_left) / self.wheel_base  # Angular velocity
        
        # Update pose using kinematic equations
        new_theta = self.theta + omega * dt
        new_x = self.x + v * np.cos(new_theta) * dt
        new_y = self.y + v * np.sin(new_theta) * dt
        
        # Update the robot's state and history
        self.set_pose(new_x, new_y, new_theta)
        
    def get_pose(self):
        """Return the current pose (x, y, theta) of the robot."""
        return self.x, self.y, self.theta

    def plot_performance(self, goal_x, goal_y):
        """
        Generates and saves all required performance plots for the report.
        """
        time_steps = range(len(self.path_x))
        
        # 1. Plot x(t) and y(t) over time
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(time_steps, self.path_x, 'r-', label='x(t)')
        plt.plot(time_steps, self.path_y, 'b-', label='y(t)')
        plt.xlabel('Time Step')
        plt.ylabel('Position (pixels)')
        plt.title('Robot Position vs. Time')
        plt.legend()
        plt.grid(True)
        
        # 2. Plot orientation θ(t) over time
        plt.subplot(2, 2, 2)
        plt.plot(time_steps, self.path_theta, 'g-')
        plt.xlabel('Time Step')
        plt.ylabel('Orientation (radians)')
        plt.title('Robot Orientation θ(t) vs. Time')
        plt.grid(True)
        
        # 3. Calculate and plot distance to goal (error) over time
        distances_to_goal = []
        for x, y in zip(self.path_x, self.path_y):
            dist = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)
            distances_to_goal.append(dist)
            
        plt.subplot(2, 2, 3)
        plt.plot(time_steps, distances_to_goal, 'm-')
        plt.xlabel('Time Step')
        plt.ylabel('Distance to Goal (pixels)')
        plt.title('Path Error: Distance to Goal vs. Time')
        plt.grid(True)
        
        # 4. Plot the robot's trajectory (path) in 2D
        plt.subplot(2, 2, 4)
        plt.plot(self.path_x, self.path_y, 'b-', label='Path')
        plt.plot(self.path_x[0], self.path_y[0], 'go', markersize=10, label='Start')
        plt.plot(goal_x, goal_y, 'ro', markersize=10, label='Goal')
        plt.xlabel('X position (pixels)')
        plt.ylabel('Y position (pixels)')
        plt.title('Robot Trajectory')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig('../results/performance_plots.png')  # Save the combined figure
        plt.show()
        
        # Save the individual plots
        print("All performance plots saved to '../results/performance_plots.png'")

def main():
    """Test the robot's movement with a simple go-to-goal algorithm."""
    # Create a robot instance starting at (50, 250), facing right (0 radians)
    robot = DifferentialDriveRobot(x=50, y=250, theta=0)
    
    # Define the goal position
    goal_x, goal_y = 400, 400
    
    # Simulation parameters
    dt = 0.1  # Time step
    max_steps = 200  # Maximum number of steps to simulate
    reached_goal = False
    
    print("Starting simulation...")
    for step in range(max_steps):
        # 1. Get the current robot position
        x, y, theta = robot.get_pose()
        
        # 2. Calculate distance and angle to the goal
        distance_to_goal = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)
        angle_to_goal = np.arctan2(goal_y - y, goal_x - x)
        
        # 3. Simple go-to-goal controller
        if distance_to_goal < 10:
            print("Goal reached!")
            reached_goal = True
            break
            
        # Calculate the angle difference (error)
        angle_error = angle_to_goal - theta
        # Normalize the angle error to [-pi, pi]
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
        
        # Set base forward velocity
        base_speed = 5.0
        # Adjust wheel speeds based on angle error
        if abs(angle_error) > 0.2:
            v_left = -np.sign(angle_error) * 2.0
            v_right = np.sign(angle_error) * 2.0
        else:  # Else, move forward while correcting slightly
            v_left = base_speed - 2.0 * angle_error
            v_right = base_speed + 2.0 * angle_error
            
        # 4. Move the robot
        robot.move(v_left, v_right, dt)
        
        # Print progress every 20 steps
        if step % 20 == 0:
            print(f"Step {step}: Position ({x:.1f}, {y:.1f}), Distance: {distance_to_goal:.1f}")
    
    if not reached_goal:
        print("Stopped after maximum steps.")
        
    # Plot the robot's path and generate performance plots
    robot.plot_performance(goal_x, goal_y)

if __name__ == "__main__":
    main()