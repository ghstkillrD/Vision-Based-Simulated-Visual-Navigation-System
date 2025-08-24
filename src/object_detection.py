import cv2
import numpy as np

def detect_apple(image_path, show_windows=True):
    """
    Detects a red apple in an image and returns its center coordinates.
    
    Args:
        image_path (str): Path to the image file.
        show_windows (bool): Whether to display debug windows. Set to False later when integrating.
    
    Returns:
        tuple: (x, y) coordinates of the apple's center, or None if not found.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    output_image = image.copy()
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # DEFINE YOUR HSV RANGE HERE (The values you found that worked!)
    lower_red = np.array([0, 100, 100])   # <- Adjust these based on your debug results
    upper_red = np.array([10, 255, 255])  # <- Adjust these based on your debug results
    
    # Create a mask
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    
    # Clean up the mask using morphological operations (removes small noise)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours are found, find the largest one
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Calculate the center of the contour
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Only consider it a valid detection if the blob is reasonably sized
        if radius > 5: 
            # Draw the detection on the output image
            cv2.circle(output_image, center, radius, (0, 255, 0), 2)
            cv2.circle(output_image, center, 3, (0, 255, 0), -1)
            
            if show_windows:
                # Display the intermediate steps for debugging
                cv2.imshow('Original Image', image)
                cv2.imshow('Mask', mask)
                cv2.imshow('Apple Detection', output_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            print(f"Apple detected at center coordinates: {center}")
            return center
    
    print("No apple detected.")
    if show_windows:
        cv2.imshow('Original Image', image)
        cv2.imshow('Mask', mask)
        cv2.imshow('Apple Detection', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return None

# Test the function on different images
if __name__ == "_main_":
    print("Testing on simulated world...")
    detect_apple('../data/simulated_world.png')
    
    # Uncomment the lines below to test on your real apple images once you have them!
    # print("\nTesting on real apple image 1...")
    # detect_apple('../data/apple_1.jpg')
    # print("\nTesting on real apple image 2...")
    # detect_apple('../data/apple_2.jpg')