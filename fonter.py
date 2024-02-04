import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

st.markdown(f"""
<div style="
  background-color: #f2f2f2;
  border-radius: 20px;
  padding: 10px;
  display: inline-block;
  margin-left: auto;
  margin-right: auto;
">
  <h1 style="color: black;"><b><i>Fonter</b></i></h1>
</div> 
""", unsafe_allow_html=True)

# Add vertical space with an empty div
st.markdown("<div style='margin-bottom: 20px; pos'></div>", unsafe_allow_html=True)

st.markdown(f"""
<div style="
  background-color: #f2f2f2;
  border-radius: 20px;
  padding: 10px;
  display: inline-block;
">
  <h3 style="color: black;"><i>Combine two fonts to produce a new font that captures the variability among both</i></h3>
</div>
""", unsafe_allow_html=True)

page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://wallpaperaccess.com/full/5823422.jpg");
  background-size: cover;
}
[data-testid="stHeader"]{
  background-color: rgba(0,0,0,0);
}
</style>
"""

st.markdown(page_element, unsafe_allow_html=True)
 
letter = st.text_input("Enter a letter:", "H")
image_width = 100
image_height = 100
 
# Create a new blank image (white background)
image_1 = Image.new("RGB", (image_width, image_height), "white")
image_2 = Image.new("RGB", (image_width, image_height), "white")
font_paths = {
    "Arial": "C:/Users/AARON/Downloads/arial.ttf",
    "Vremya": "C:/Users/AARON/Downloads/vremya.ttf",
    "Arial Narrow": "C:/Users/AARON/Downloads/Arialn.ttf",

}

font1 = st.selectbox("Select Font 1", ("Arial","Vremya","Arial Narrow"))
font2 = st.selectbox("Select Font 2", ("Arial","Vremya","Arial Narrow"))

font1_path = font_paths[font1]
font2_path = font_paths[font2]
font1 = ImageFont.truetype(font1_path ,size=50)
font2 = ImageFont.truetype(font2_path, size=50)
 
# Create a drawing context
draw_1 = ImageDraw.Draw(image_1)
draw_2 = ImageDraw.Draw(image_2)
 
# Set the text to be drawn (the letter entered by the user)
text = letter
 
# Get the size of the text
text_width, text_height = draw_1.textsize(text, font=font1)
 
# Calculate the position to center the text
x = (image_width - text_width) / 2
y = (image_height - text_height) / 2
 
# Draw the text on the images
draw_1.text((x, y), text, fill="black", font=font1)
draw_2.text((x, y), text, fill="black", font=font2)
 
# Save the images
image_1.save("L11.png")
image_2.save("L22.png")
 
# Load images
image1 = cv2.imread('L11.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('L22.png', cv2.IMREAD_GRAYSCALE)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Create a subplot with 1 row and 2 columns
axes[0].imshow(image1,cmap = 'gray')
axes[0].set_title('Font 1 Letter')
axes[0].axis('off')
axes[1].imshow(image2,cmap='gray')
axes[1].set_title('Font 2 Letter')
axes[1].axis('off')
 
# Display the plot
st.pyplot(fig)
 
# Ensure images have the same dimensions
image1 = cv2.resize(image1, (100, 100))
image2 = cv2.resize(image2, (100, 100))
 
# Convert images to arrays
image1_array = np.array(image1).flatten()
image2_array = np.array(image2).flatten()
 
# Combine images into one dataset
images_dataset = np.vstack((image1_array, image2_array))
 
# Apply PCA
pca = PCA(n_components=1)  # You can change the number of components as needed
pca.fit(images_dataset)
 
# Retrieve eigenfaces
eigenfaces = pca.components_
for i, eigenface in enumerate(eigenfaces, 1):
    eigenface_image = eigenface.reshape(image1.shape)
# Apply Laplacian operator for edge detection
laplacian = cv2.Laplacian(eigenface_image, cv2.CV_64F)
sobel_x = cv2.Sobel(eigenface_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(eigenface_image, cv2.CV_64F, 0, 1, ksize=3)
 
# Compute the gradient magnitude
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
# Normalize the gradient magnitude to [0, 255]
gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
 
# Overlay edges on the original image for visualization
overlay = cv2.addWeighted(eigenface_image, 0.7, gradient_magnitude, 0.3, 0)
blurred = cv2.GaussianBlur(gradient_magnitude, (0, 0), 3)
sharpened = cv2.addWeighted(gradient_magnitude, 1, blurred, -0.5, 0)
# Convert the sharpened image to 8-bit unsigned integer format
sharpened_uint8 = cv2.convertScaleAbs(sharpened)
 
inverted_sharpened = 255 - sharpened
 
 
# Convert the sharpened image to binary using thresholding
_, binary_image = cv2.threshold(sharpened_uint8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
 
 
# Find contours on the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
# Create a white background image
filled_shape = np.ones_like(eigenface_image) * 255
 
# Draw contours on the white image
cv2.drawContours(filled_shape, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
 
# Overlay filled shape on the original eigenface image for visualization
overlay = cv2.addWeighted(eigenface_image, 0.7, filled_shape, 0.3, 0)
threshold = 0.1  # Adjust this value as needed (0 is black, 255 is white)
 
# Create a binary image where black pixels stay black and others become white
binary_image = cv2.threshold(sharpened, threshold, 255, cv2.THRESH_BINARY)[1]
 
# Optional: Combine with the original image for blended effect
blended_image = cv2.addWeighted(sharpened, 0.7, binary_image, 0.3, 0)
 
# Assuming you have your images in 'sharpened' and 'filled_shape' variables
fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Create a subplot with 1 row and 2 columns
 
# Plot the sharpened image
axes[0].imshow(sharpened,cmap = 'gray')
axes[0].set_title('Sharpened EigenFace')
axes[0].axis('off')
 
# Plot the filled shape image
axes[1].imshow(255-blended_image,cmap='gray')
axes[1].set_title('Thresholded Font')
axes[1].axis('off')
 
# Display the plot
st.pyplot(fig)  # Displaying the plot using Streamlit
