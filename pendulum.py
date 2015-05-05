'''
Created April 2015 by Evan CHOW and Gabriel HUANG
for Seung - COS 598C - Princeton University
'''

from Tkinter import Tk, Canvas
import math
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt

WIDTH = 500
HEIGHT = 300
BOARD_OFFSET_BOTTOM = -10
BOARD_WIDTH = 40
BOARD_HEIGHT = 13

FPS = 60
ALPHA_LIM = math.pi/3.
HYSTERESIS = 0.01
REWARD_HYSTERISIS = 0.02

class Display:
    def __init__(self, fps=FPS, width=WIDTH, height=HEIGHT,
        board_offset_bottom=BOARD_OFFSET_BOTTOM,
        board_width=BOARD_WIDTH,
        board_height=BOARD_HEIGHT):
        self.root=Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.width = width
        self.height = height
        self.canvas=Canvas(self.root, bg="black",width=width,height=height)
        self.board_width = board_width
        self.board_height = board_height
        self.board_offset_bottom = board_offset_bottom
        self.canvas.pack()
        self.fps = fps
        self.controllers = []
        #For reset
        self.root.bind("<space>", lambda e:self.reset_all())
        self.root.bind("<Escape>", lambda e:self.root.destroy())
    def run(self):
        self.root.after(1000//self.fps, self.loop)
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.root.destroy()
    def loop(self):
        actions = [controller.get_action(self.model) for controller in self.controllers]
        self.model.update(1./self.fps, actions)
        self.draw()
        self.root.after(1000//self.fps, self.loop)
    def draw(self):
        self.canvas.delete('all')
        self.board = self.canvas.create_rectangle(
            self.model.x()-self.board_width/2,
            self.board_offset_bottom+self.height-self.board_height,
            self.model.x()+self.board_width/2,
            self.board_offset_bottom+self.height, fill="green")
        self.pendulum = self.canvas.create_line(
            self.model.x(),
            self.board_offset_bottom+self.height-self.board_height,
            self.model.x()+self.model.arm_length*math.sin(self.model.alpha()),
            self.board_offset_bottom+self.height-self.board_height-self.model.arm_length*math.cos(self.model.alpha()),
            fill="blue", width=20)
    def attach_model(self, model):
        self.model = model
        self.draw()
    def attach_controller(self, controller):
        self.controllers.append(controller)
    def reset_all(self):
        self.model.randomize()


PEND_ARM_LENGTH = 80
class Pendulum:
    # States (x,alpha,dx,dalpha)
    # Actions (nothing, left thrust, right thrust)
    g = 1000 
    thrust = 15000
    arm_length = PEND_ARM_LENGTH # pixels
    drag = 5.
    drag_alpha = 1.5
    def __init__(self, state=np.zeros(4)):
        if state is None:
            self.randomize()
        else:
            self.state = state
    def x(self):
        return self.state[0]
    def v_x(self):
        return self.state[1]
    def alpha(self):
        return self.state[2]
    def v_alpha(self):
        return self.state[3]
    def randomize(self):
        self.state = self.sample_state()
    def sample_state(self, border=0, alpha_max=math.pi/2, v_x_max=10, v_alpha_max=math.pi/10):
        return np.array([random.uniform(border,WIDTH-border),
                         random.uniform(-v_x_max,v_x_max),
                        alpha_max*random.uniform(-1,1), 
                            random.uniform(-v_x_max,v_x_max)])
    def update(self, dt, actions=[]):
        self.state = self.peek_update(self.state, dt, actions)
    def peek_update(self, state, dt, actions=[]):
        '''
        Implements physical model
        '''
        # Thrust and drag
        a_x = 0
        for action in actions:      
            if action=='left':
                this_a_x = -self.thrust
            elif action=='right':
                this_a_x = +self.thrust
            else:
                this_a_x = 0
            a_x += this_a_x
        a_x -= self.drag*state[1]
        v_x = state[1] + a_x * dt
        x = state[0] + v_x * dt
        a_alpha = -self.drag_alpha * state[3] + (self.g*math.sin(self.alpha()) - a_x * math.cos(state[2]))/self.arm_length
        v_alpha = self.v_alpha() + a_alpha * dt
        alpha = self.alpha() + v_alpha * dt
        # Pendulum angle limit
        if alpha > ALPHA_LIM:
            alpha = ALPHA_LIM * (1-HYSTERESIS)
            v_alpha = 0.
        elif alpha < -ALPHA_LIM:
            alpha = -ALPHA_LIM * (1-HYSTERESIS)
            v_alpha = 0.
        # Board x position limit
        if x > WIDTH:
            x = WIDTH * (1 - HYSTERESIS)
            v_x = 0.
        elif x < 0:
            x = WIDTH * HYSTERESIS
            v_x = 0.
        return np.array([x, v_x, alpha, v_alpha])
    def __repr__(self):
        return 'x:{} alpha:{} dx:{} dalpha{}'.format(self.x(), self.alpha(), self.v_x(), self.v_alpha())
       


class Controller:
    '''
    Abstract Class for controlling the force input on the board
    Given a current state return an action to take
    '''
    def get_action(self, model):
        '''
        To be overloaded by derived classes
        returns 'none', 'left' or 'right'
        in function of state
        Typically RL-algorithms will call model.update()
        '''
        pass
    
class RlController(Controller):
    '''
    Abstract Class for board-controlling RL algorithms
    Dont forget to overlaod get_action
    '''
    def learn_values(self, model, cycles):
        '''Given the model, learn to approximate the value of each state'''        
        pass
    
class KeyController(Controller):
    '''Human controller (uses arrows of keyboard)'''
    def __init__(self, display):
        self.key = 'none'
        display.root.bind("<KeyPress>", self.set_key)
        display.root.bind("<KeyRelease>", self.reset_key)
    def reset_key(self, e):  
        self.key = 'none'
    def set_key(self, e):
        if e.keysym == 'Right':
            self.key = 'right'
        elif e.keysym == 'Left':
            self.key = 'left'
    def get_action(self, model):
        return self.key

class PidController(Controller):
    '''Simple PID controller (actually just a PD)'''
    def __init__(self, p=20., d=0.5):
        self.p = p
        self.d = d
    def get_action(self, model):
        order = self.p*model.alpha()+self.d*model.v_alpha()
        return pwm(order)

def pwm(direction):
    '''
    direction between -1 and 1
    '''
    if direction>0. and random.uniform(0,1)<direction:
        return 'right'
    elif direction<0. and random.uniform(0,1)<-direction:
        return 'left'
    else:
        return 'none'
        
        
class Normalizer:
    def __init__(self, regressor, scale):
        self.regressor = regressor
        self.scale = scale
    def fit(self, X, y):
        X2 = X / self.scale
        print X2[0]
        return Normalizer(self.regressor.fit(X2, y), self.scale)
    def predict(self, X):
        X2 = X / self.scale
        return self.regressor.predict(X2)
        
class RlLinearController(RlController):
    def get_action(self, model):
        # Select action with highest value from here
        best_action,best_value = 'none',float('-inf')
        for action in ['none','left','right']:
            new_state = model.peek_update(model.state, 1./FPS, actions=[action])
            value = self.learner.predict(new_state)[0]
            print '{} --> {}'.format(action, value)
            if value>best_value:
                best_value = value   
                best_action = action
        #print '{} ----> Value {} / Reward {}'.format(best_action, best_value, reward(new_state))
        return best_action
    def learn_values(self, model, learner_generator, reward, gamma, num_samples, iterations):
        # Sample states    
        global data
        data = np.zeros((num_samples, 4))
        # Estimated values for sampes
        estimated_values = np.zeros(num_samples)
        for it in range(iterations):        
            # Sample states
            for i in range(num_samples):
                data[i] = model.sample_state()
            # For each state in samples
            for i, state in enumerate(data):
                # Find best action
                estimated_values[i] = float('-inf')
                for action in ['none','left','right']:
                    new_state = model.peek_update(state, 1./FPS, actions=[action])
                    value = reward(state) + (gamma * self.learner.predict(new_state) if it else 0.)
                    if value>estimated_values[i]:
                        estimated_values[i] = value
            # Relearn estimates
            print 'iteration {}/{}'.format(it+1, iterations)
            self.learner = learner_generator.fit(data, estimated_values)
        
# Reward function
def reward(state):
    if state[0]<=WIDTH*REWARD_HYSTERISIS or state[0]>=WIDTH*(1-REWARD_HYSTERISIS):
        return -1
    if abs(state[2])>=ALPHA_LIM*(1-REWARD_HYSTERISIS):
        return -1
    return 0
        
# Code       
display = Display()
pendulum = Pendulum()
pendulum.randomize()
# key_controller = KeyController(display)
# pid_controller = PidController()
display.attach_model(pendulum)
#display.attach_controller(pid_controller) # comment to disable pid controller
# display.attach_controller(key_controller)

rl_linear_controller = RlLinearController()
display.attach_controller(rl_linear_controller)
regressor = Normalizer(SVR(C=0.01), np.array([WIDTH, WIDTH*10e1 , math.pi*0.6, math.pi*0.6*10e1]))
rl_linear_controller.learn_values(pendulum, regressor,#RandomForestRegressor(n_estimators=64), 
                                  reward, gamma=0.995, num_samples=1000, iterations=32)
#%% Plot values
values = np.array([[rl_linear_controller.learner.predict([x,0,alpha,0])[0] 
for x in np.arange(0,WIDTH,WIDTH/50)] for alpha in np.arange(-math.pi/2,math.pi/2,math.pi/50)])

plt.subplot(131)
plt.imshow(values,interpolation='nearest')
plt.xlabel('x')
plt.ylabel('alpha')

plt.subplot(132)
plt.plot(np.arange(-math.pi/2,math.pi/2,math.pi/50),
    np.array([rl_linear_controller.learner.predict([300,0,alpha,0])[0] for alpha in np.arange(-math.pi/2,math.pi/2,math.pi/50)]))
plt.xlabel('alpha (rad)')
plt.ylabel('V(alpha,x=300)')


plt.subplot(133)
plt.plot(np.arange(0,WIDTH,WIDTH/50),
    np.array([rl_linear_controller.learner.predict([x,0,0,0])[0] for x in np.arange(0,WIDTH,WIDTH/50)]))
plt.xlabel('x (pix)')
plt.ylabel('V(x,alpha=0)')

#%%
display.run()