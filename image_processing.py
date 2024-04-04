from PIL import Image
import numpy as np

# Load the image
image = Image.open('path_to_your_image.jpg')
image = image.convert('L')  # Convert to grayscale for simplicity

# Convert the image to a numpy array
image_matrix = np.array(image)

# Perform an operation (e.g., invert colors)
inverted_matrix = 255 - image_matrix

# Define a simple scaling matrix (for demonstration, scales by a factor of 2)
# In a realistic scenario, more complex matrices and operations would be used for scaling
scaling_matrix = np.array([[2, 0], [0, 2]])

# Prepare the image matrix for matrix multiplication by adding a column of ones (homogeneous coordinates)
homogeneous_matrix = np.ones((image_matrix.shape[0], image_matrix.shape[1]+1))
homogeneous_matrix[:, :-1] = inverted_matrix

# Perform matrix multiplication (this is a simplified example, actual image scaling is more complex)
# Note: For a real scaling operation, interpolation methods and more complex transformations would be needed
scaled_matrix = np.dot(homogeneous_matrix, scaling_matrix.T)

# For simplicity, we're only using the first two columns after scaling (ignoring the homogeneous coordinate)
result_matrix = scaled_matrix[:, :-1]

# Clip values to valid range and convert back to an image
result_image = Image.fromarray(np.clip(result_matrix, 0, 255).astype('uint8'))

# Save the resulting image
result_image.save('result_image.jpg')

print("Operation complete. The resulting image has been saved as 'result_image.jpg'.")
