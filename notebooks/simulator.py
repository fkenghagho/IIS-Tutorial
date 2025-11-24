from jyro.simulator import *
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import LogNorm
from scipy.interpolate import CubicSpline
import random
import math

class Speaker(Speech):
    def speak(self, robot, canvas, text_to_say):
        text_to_say=str(text_to_say)
        #canvas.drawText(5, 5, text_to_say)
        canvas.drawText(robot.getPose()[0]+0.2, robot.getPose()[1]+0.2, text_to_say, fill="red")

class StereoCamera(Camera):
    def __init__(self,index,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type='stereo_camera'+str(index)
        self.bx = [ .14, .06, .06, .14] # front camera
        self.by = [-.06, -.06, .06, .06]
    
    def draw(self, robot, canvas):
        
        a90 = robot._ga + PIOVER2 # angle is 90 degrees off for graphics
        cos_a90 = math.cos(a90)
        sin_a90 = math.sin(a90)
        xy = map(lambda x, y: (robot._gx + x * cos_a90 - y * sin_a90,
                               robot._gy + x * sin_a90 + y * cos_a90),
                 self.bx, self.by)
        xy = list(xy)
        canvas.drawPolygon(xy, fill="black")
        

class Sonar(Pioneer16Sonars):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.std0=random.random()/160.0
        self.std15=random.random()/130.0
        self.noisy_scan=self.scan.copy()
        
        
    
    def getData(self):
        self.noisy_scan=self.scan.copy()
        self.noisy_scan[0]=np.abs(self.noisy_scan[0]+np.random.normal(0.0, self.std0))
        self.noisy_scan[15]=np.abs(self.noisy_scan[15]+np.random.normal(0.0, self.std15))
        return self.noisy_scan[:] # copy
