############################################################
#
# The convolutional network that implements Q-Learning on
# the pole balancing game.
# 
# Input: a stack of images
# 
############################################################

import Image, time
import process_image

# run screengrab once to bring the game window to the foreground
process_image.screengrab()

# Grab a stack of 10 screens, each 1/10 of a second apart
# You can display these to make sure you grabbed them correctly (shows
# actual movement of the board)
screens = []
for i in xrange(10):
    screens.append(process_image.screengrab())
    time.sleep(0.1)








#############################################################

# save images
# for i in xrange(10):
#     img = Image.fromarray(screens[i])
#     img.save('screengrab{}.png'.format(i))

import code
code.interact(local=locals())