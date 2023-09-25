
from PIL import Image

# Load the image
img_path = './concat_guassian_noise.png'
img = Image.open(img_path)

# Resize the image to 64x64
img_resized = img.resize((64, 64), Image.ANTIALIAS)

# Save the resized image
img_resized.save('./concat_guassian_noise_64.png')

