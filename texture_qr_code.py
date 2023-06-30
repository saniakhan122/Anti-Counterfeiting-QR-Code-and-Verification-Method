import numpy as np
from scipy.ndimage import zoom
from PIL import Image
import cv2
from pyzbar.pyzbar import decode

def generate_gaussian_texture(height, width, mean, std_dev):
    # Generate a random matrix with Gaussian distribution
    random_matrix = np.random.normal(mean, std_dev, size=(height, width))
    

    # Normalize the values between 0 and 255
    normalized_matrix = cv2.normalize(random_matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return normalized_matrix

# Specify the desired size and parameters for the Gaussian texture
height = 512
width = 512
mean = 120  # Mean of the Gaussian distribution
std_dev = 50  # Standard deviation of the Gaussian distribution

# Generate the Gaussian texture pattern
gaussian_texture = generate_gaussian_texture(height, width, mean, std_dev)

# Display the generated texture pattern
cv2.imshow("Gaussian Texture", gaussian_texture)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the generated texture pattern as an image
cv2.imwrite("gaussian_texture_pattern.png", gaussian_texture)
#Step 2
def bilinear_interpolation(img, x, y):
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + 1
    y2 = y1 + 1

    # Check if the coordinates are within the image bounds
    if x2 >= img.shape[1]:
        x2 = img.shape[1] - 1
    if y2 >= img.shape[0]:
        y2 = img.shape[0] - 1

    # Calculate the weights
    dx = x - x1
    dy = y - y1

    # Perform bilinear interpolation
    R1 = (x2 - x) * img[y1, x1] + (x - x1) * img[y1, x2]
    R2 = (x2 - x) * img[y2, x1] + (x - x1) * img[y2, x2]
    interpolated_value = (y2 - y) * R1 + (y - y1) * R2

    return interpolated_value

# Load the texture pattern image
texture_pattern = cv2.imread("gaussian_texture_pattern.png", cv2.IMREAD_GRAYSCALE)

# Specify the scale factors for interpolation
scale_factor_x = 2  # Scale factor along the x-axis
scale_factor_y = 2  # Scale factor along the y-axis

# Compute the new dimensions after scaling
new_width = int(texture_pattern.shape[1] * scale_factor_x)
new_height = int(texture_pattern.shape[0] * scale_factor_y)

# Create a new blank image for the interpolated texture pattern
interpolated_texture = np.zeros((new_height, new_width), dtype=np.uint8)

# Perform bilinear interpolation to generate the new texture pattern
for y in range(new_height):
    for x in range(new_width):
        src_x = x / scale_factor_x
        src_y = y / scale_factor_y
        interpolated_value = bilinear_interpolation(texture_pattern, src_x, src_y)
        interpolated_texture[y, x] = interpolated_value

# Display the interpolated texture pattern
cv2.imshow("Interpolated Texture", interpolated_texture)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the interpolated texture pattern as an image
cv2.imwrite("interpolated_texture_pattern.png", interpolated_texture)


# Step 3: Adopt halftone treatment
def error_diffusion_halftone(image):
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert the image to a NumPy array
    img_array = np.array(image)
    
    # Define the error diffusion matrix (1D)
    error_matrix = np.array([0, 0, 0.4375, 0.1875])
    
    # Iterate over each pixel in the image
    height, width = img_array.shape
    for y in range(height):
        for x in range(width):
            # Get the current pixel value
            pixel = img_array[y, x]
            
            # Calculate the quantization error
            error = pixel - (pixel > 127) * 255
            
            # Distribute the error to neighboring pixels
            for i, d in enumerate(error_matrix):
                if x+i < width:
                    img_array[y, x+i] += error * d
            
            # Update the current pixel value
            img_array[y, x] = (pixel > 127) * 255
    
    # Create a PIL Image object from the halftoned array
    halftoned_image = Image.fromarray(img_array)
    
    return halftoned_image

# Load the texture image
texture_image = Image.open('interpolated_texture_pattern.png')

# Apply error diffusion halftoning
halftoned_texture = error_diffusion_halftone(texture_image)
scale_factor_x
# Save the halftoned image
halftoned_texture.save('halftoned_texture.png')


# Step 4: Refine QR code
def refine_qr_code(qr_code, reduction_size):
    # Convert the QR code image to a NumPy array
    qr_array = np.array(qr_code)
    
    # Calculate the reduction size for local pixel size
    a = reduction_size + 1
    
    # Refine the areas of the QR code
    refined_qr_array = np.copy(qr_array)
    
    for y in range(qr_array.shape[0]):
        for x in range(qr_array.shape[1]):
            if np.all(qr_array[y, x]) == 0:  # QR code module     
                 refined_qr_array[y, x] = qr_array[max(y - a, 0):min(y + a + 1, qr_array.shape[0]), max(x - a, 0):min(x + a + 1, qr_array.shape[1])].mean()
            elif np.all(qr_array[y, x] == 255):  # Local center area
                refined_qr_array[y, x] = 255
    
    # Create a PIL Image object from the refined QR code array
    refined_qr_code = Image.fromarray(refined_qr_array)
    
    return refined_qr_code

# Load the QR code image
qr_code_image = Image.open("C:/Users/SANIA/Downloads/QR-DN1.0/QR/train/ht2.jpg")

# Define the reduction size for local pixel size
reduction_size = 1  # Adjust this value based on the desired reduction size

# Refine the QR cod
refined_qr_code = refine_qr_code(qr_code_image, reduction_size)

# Save the refined QR code image
refined_qr_code.save('refined_qr_code.png')

def combine_texture_patterns(refined_qr_code, texture_patterns):
    # Resize the texture patterns to match the size of the refined QR code
    resized_texture_patterns = texture_patterns.resize(refined_qr_code.shape[::-1])

    # Convert the refined QR code image and resized texture patterns to NumPy arrays
    qr_array = np.array(refined_qr_code)
    texture_array = np.array(resized_texture_patterns)

    # Combine the texture patterns and refined QR code
    combined_array = np.where(qr_array == 0, qr_array, texture_array)

    # Create a PIL Image object from the combined array
    combined_image = Image.fromarray(combined_array)

    return combined_image

# Load the refined QR code image
refined_qr_code = cv2.imread('refined_qr_code.png', 0)

# Load the texture patterns image
texture_patterns = Image.open('halftoned_texture.png')

# Combine the texture patterns and refined QR code
combined_image = combine_texture_patterns(refined_qr_code, texture_patterns)

# Convert the combined image to grayscale
combined_image_gray = combined_image.convert('L')

# Convert the combined image to OpenCV format
combined_image_cv = np.array(combined_image_gray)

# Decode the QR code
decoded_qr_codes = decode(combined_image_cv)


if len(decoded_qr_codes) > 0:
    qr_data = decoded_qr_codes[0].data.decode('utf-8')
    print("Decoded QR code data:", qr_data)
else:
    # Handle the case when no QR codes were decoded
    print("QR code not found")
combined_image.save('combined_image.png')

