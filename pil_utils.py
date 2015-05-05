import win32gui
import Image

def RaiseWindowNamed(nameRe):
    """ Bring a window to the front based on its name. """
    # start by getting a list of all the windows:
    cb = lambda x,y: y.append(x)
    wins = []
    win32gui.EnumWindows(cb,wins)
    # now check to see if any match our regexp:
    tgtWin = -1
    for win in wins:
        txt = win32gui.GetWindowText(win)
        if nameRe == txt:
            tgtWin=win
            break
    if tgtWin>=0:
        win32gui.SetForegroundWindow(tgtWin)