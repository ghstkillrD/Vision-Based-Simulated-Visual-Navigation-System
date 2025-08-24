# This is the main script to integrates object detection with robot simulation.
from object_detection import detect_apple
from robot_simulation import DifferentialDriveRobot
import cv2
import numpy as np

def run_full_simulation(world_image_path):
    """
    Runs the complete integrated simulation:
    1. Detect the apple in the world image to get the goal coordinates.
    2. Simulate the robot navigating to those coordinates.
    3. Generate and save all performance plots.
    """
    print("="*50)
    print("STARTING INTEGRATED SIMULATION")
    print("="*50)
    
    print("\n1. Detecting apple in the world...")
    goal_point = detect_apple(world_image_path, show_windows=True)
    
    if goal_point is None:
        print("ERROR: Could not find the apple. Cannot run simulation.")
        return
    
    goal_x, goal_y = goal_point
    print(f"Apple detected at: ({goal_x}, {goal_y})")
    
    print("\n2. Starting robot simulation to the detected goal...")
    # Create a robot instance starting at (50, 250) and facing right (0 radians)
    robot = DifferentialDriveRobot(x=50, y=250, theta=0)
    
    # Simulation parameters
    dt = 0.1
    max_steps = 300
    reached_goal = False
    
    for step in range(max_steps):
        x, y, theta = robot.get_pose()
        
        # Calculate distance and angle to the goal
        distance_to_goal = np.sqrt((goal_x - x)*2 + (goal_y - y)*2)
        angle_to_goal = np.arctan2(goal_y - y, goal_x - x)
        
        # Simple go-to-goal controller
        if distance_to_goal < 15:  # If we are close to the goal, stop
            print("Goal reached!")
            reached_goal = True
            break
            
        # Calculate the angle difference
        angle_error = angle_to_goal - theta
        # Normalize the angle error
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
        
        # Set the base forward velocity
        base_speed = 5.0
        # Adjusting wheel speeds
        if abs(angle_error) > 0.2:  # If error is large, turn in place
            v_left = -np.sign(angle_error) * 2.0
            v_right = np.sign(angle_error) * 2.0
        else:
            v_left = base_speed - 2.0 * angle_error
            v_right = base_speed + 2.0 * angle_error
            
        robot.move(v_left, v_right, dt)
        
        if step % 25 == 0:
            print(f"Step {step}: Position ({x:.1f}, {y:.1f}), Distance: {distance_to_goal:.1f}")
    
    if not reached_goal:
        print(f"Stopped after {max_steps} steps. Final distance: {distance_to_goal:.1f}")
        
    print("\n3. Generating performance plots and visualization...")
    # Create the performance plots
    robot.plot_performance(goal_x, goal_y)
    
    # Create visualization image
    world_image = cv2.imread(world_image_path)
    for i, (x, y) in enumerate(zip(robot.path_x, robot.path_y)):
        cv2.circle(world_image, (int(x), int(y)), 2, (255, 0, 0), -1)  # Blue path
        if i == len(robot.path_x) - 1:  # Draw last position larger
            cv2.circle(world_image, (int(x), int(y)), 8, (0, 0, 255), -1)  # Red final position
    
    # Draw start and goal
    cv2.circle(world_image, (int(robot.path_x[0]), int(robot.path_y[0])), 8, (0, 255, 0), -1)  # Green start
    cv2.circle(world_image, (goal_x, goal_y), 8, (0, 0, 255), -1)  # Red goal
    
    # Save the final visualized world
    output_path = '../results/final_navigation_result.png'
    cv2.imwrite(output_path, world_image)
    print(f"Final navigation visualization saved to: {output_path}")
    
    print("\n" + "="*50)
    print("SIMULATION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    run_full_simulation('../data/simulated_world.png')