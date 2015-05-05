# Process the image into input for the convnet.
# Should have it at the foreground I think.
# Note that you can just run this every few frames or seconds since
# they didn't even use every frame in the Atari paper.
import ImageGrab, Image, time, win32gui

import numpy as np
import winGuiAuto
from pil_utils import * # some helper functions

def screengrab():
    """ Get the current screen, convert it into a data array,
        and return the array. """

    # Get the pole balancing window (with name "tk" for TKinter) and
    # bring it to the front. Take a screenshot.
    RaiseWindowNamed("tk")
    # time.sleep(0.5) # make sure gets to front
    rect = win32gui.GetWindowPlacement(winGuiAuto.findTopWindow("tk"))[-1]
    image = ImageGrab.grab(rect)

    # Extract pixel data from image. Converts to B&W since we don't need RGB.
    # http://choorucode.com/2014/01/16/
    # how-to-convert-between-numpy-array-and-pil-image/
    data_arr = np.array(image.convert("L")) # shape: (height x width)

    # Save recreated image: QA to make sure the img -> array works
    # img = Image.fromarray(data_arr)
    # img.save('recreated.png')

    return data_arr