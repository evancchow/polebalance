# Test screengrab for a window.
# Apparently this works even when the window is not at the foreground!
import winGuiAuto
import win32gui
import ImageGrab

sample = winGuiAuto.findTopWindow("tk")
rect = win32gui.GetWindowPlacement(sample)[-1]
print rect
image = ImageGrab.grab(rect)

# note: does not open directly from terminal.
# need to convert to pixel values directly to feed into convnet
image.save("sample.jpg", "JPEG")