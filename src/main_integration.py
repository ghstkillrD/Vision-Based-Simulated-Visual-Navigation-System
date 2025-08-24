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
    
    # Step 1: Object Detection
    print("\n1. Detecting apple in the world...")
    goal_point = detect_apple(world_image_path, show_windows=True)
    
    if goal_point is None:
        print("ERROR: Could not find the apple. Cannot run simulation.")
        return
    
    goal_x, goal_y = goal_point
    print(f"Apple detected at: ({goal_x}, {goal_y})")
    
    # Step 2: Robot Simulation
    print("\n2. Starting robot simulation to the detected goal...")
    # Create a robot instance starting at (50, 250), facing right (0 radians)
    robot = DifferentialDriveRobot(x=50, y=250, theta=0)
    
    # Simulation parameters
    dt = 0.1  # Time step
    max_steps = 300  # Maximum number of steps to simulate
    reached_goal = False
    
    for step in range(max_steps):
        # Get the current robot position
        x, y, theta = robot.get_pose()
        
        # Calculate distance and angle to the goal
        distance_to_goal = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)
        angle_to_goal = np.arctan2(goal_y - y, goal_x - x)
        
        # Simple go-to-goal controller
        if distance_to_goal < 15:  # If we are close to the goal, stop
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
        else:  # Else, move forward while correcting
            v_left = base_speed - 2.0 * angle_error
            v_right = base_speed + 2.0 * angle_error
            
        # Move the robot
        robot.move(v_left, v_right, dt)
        
        # Print progress every 25 steps
        if step % 25 == 0:
            print(f"Step {step}: Position ({x:.1f}, {y:.1f}), Distance: {distance_to_goal:.1f}")
    
    if not reached_goal:
        print(f"Stopped after {max_steps} steps. Final distance: {distance_to_goal:.1f}")
        
    # Step 3: Generate Results
    print("\n3. Generating performance plots and visualization...")
    # Generate the four performance plots
    robot.plot_performance(goal_x, goal_y)
    
    # Create a final visualization
    world_image = cv2.imread(world_image_path)
    for i, (x, y) in enumerate(zip(robot.path_x, robot.path_y)):
        # Draw the robot's path on the world image
        cv2.circle(world_image, (int(x), int(y)), 2, (255, 0, 0), -1)  # Blue path
        # Draw the robot's current position
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
    # Run the complete simulation
    run_full_simulation('../data/3_100.jpg')
    