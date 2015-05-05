############################################################
#
# The convolutional network that implements Q-Learning on
# the pole balancing game.
# 
# Input: a stack of images
# 
############################################################

import Image
import process_image

# Save recreated image: QA to make sure the img -> array works
currscreen = process_image.screengrab()
img = Image.fromarray(currscreen)
img.save('recreated.png')