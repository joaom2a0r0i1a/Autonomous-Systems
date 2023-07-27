#!/usr/bin/env python3

# RUN: python3 micro_simulator.py MAP_PATH MP4_PATH

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anmtn
from matplotlib.collections import LineCollection

from map import Map
from odom import Odom
from scan import Scan
from particle_filter import ParticleFilter



np.set_printoptions(threshold = np.inf, precision = 5)

# Number of particles
N = 400
SAVE = True
KIDNAP = False
GAUSSIAN = False

class MicroSimulator():

    def __init__(self, file: str):
        
        # Information variables
        self.map = None
        self.odom = None
        self.scan = None
        self.position = np.zeros(3)
        self.estimate = np.zeros(3)
        self.pf = ParticleFilter(N)

        # Plot variables
        self.image = None
        self.p_position = None
        self.p_position_rot = None
        self.p_scan = None
        self.p_estimate = None
        self.p_estimate_rot = None
        self.p_particles = None

        self.loadmap(file, 0.5)
        self.loadplot()
        
    def loadmap(self, file: str, resolution: int):
        """
        Load the map in file (.png) with resolution [m/pixel] and initialize the particles
        """
        
        # 0 -> White, 255 -> Black
        data = np.invert(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY))
        width = data.shape[0]
        height = data.shape[1]
        x_range = (0, (width - 1)  * resolution)
        y_range = (0, (height - 1) * resolution)

        self.map = Map(data, width, height, resolution, x_range, y_range)

        print(f"Loading map from file \"{file}\"\n")
        print(str(self.map))

        # Initialize the particles
        print(f"Initializing {N} particles\n")
        if GAUSSIAN:
            pose_x, pose_y = self.map.coord_to_pose(x = 10, y = 90)
            mean = np.array([pose_x, pose_y, -0.5 * np.pi])
            std = np.array([1, 1, 0.25 * np.pi])
            self.pf.create_gaussian_particles(mean = mean, std = std, N = N, map = self.map)
        else:
            self.pf.create_uniform_particles(x_range, y_range, (-1 * np.pi, np.pi), N, self.map)
            pose_x, pose_y = self.map.coord_to_pose(x = 10, y = 90)
            self.pf.particles[0, :] = np.array([pose_x, pose_y, - 0.5 * np.pi])

    def loadplot(self):
        """
        Load the figure with the different plots
        """

        fig = plt.figure()
        self.image = fig.add_subplot(111)
        self.image.set_title("Simulation")

        # Map
        self.image.imshow(self.map.data, origin = 'lower', cmap = "Greys", interpolation = "bilinear")

        # Trajectory
        if KIDNAP:
            self.image.plot([10, 10], [90, 60], c = "red", linestyle = 'dashed', markersize = 1, alpha = 0.5, label = 'trajectory')
            self.image.plot([10, 26], [10, 10], c = "red", linestyle = 'dashed', markersize = 1, alpha = 0.5)
            self.image.plot([26, 26], [10, 34], c = "red", linestyle = 'dashed', markersize = 1, alpha = 0.5)
            self.image.plot([10], [90], c = "black", marker = "o", markersize = 5, label = 'start')
            self.image.plot([26], [34], c = "black", marker = "x", markersize = 5, label = 'finish')
            self.image.plot([10], [60], c = "grey", marker = "x", markersize = 5, label = 'kidnapped')
            self.image.plot([10], [10], c = "grey", marker = "o", markersize = 5, label = 'new position')
        else: 
            self.image.plot([10, 10], [90, 10], c = "red", linestyle = 'dashed', markersize = 1, alpha = 0.5, label = 'trajectory')
            self.image.plot([10, 26], [10, 10], c = "red", linestyle = 'dashed', markersize = 1, alpha = 0.5)
            self.image.plot([26, 26], [10, 34], c = "red", linestyle = 'dashed', markersize = 1, alpha = 0.5)
            self.image.plot([10], [90], c = "black", marker = "o", markersize = 5, label = 'start')
            self.image.plot([26], [34], c = "black", marker = "x", markersize = 5, label = 'finish')

        # Position, estimate and particles
        self.p_position = self.image.plot([], [], "go", markersize = 3, label = 'real position')
        self.p_position_rot = self.image.plot([], [], "g")
        self.p_estimate = self.image.plot([], [], "bo", markersize = 3, label = 'estimate')
        self.p_estimate_rot = self.image.plot([], [], "b")
        self.p_particles = self.image.scatter([], [], s = 5, c = 'red', label = "particles")

        # Plot appearance
        self.image.legend(loc = 1, prop = {'size': 5})
        self.image.xaxis.set_ticklabels([0, 0, 10, 20, 30, 40, 50])
        self.image.yaxis.set_ticklabels([0, 0, 10, 20, 30, 40, 50])
        self.image.set_xlabel(xlabel = "x [m]" )
        self.image.set_ylabel(ylabel = "y [m]" )

    def setInitialPosition(self, x: float, y: float, theta: float):
        """
        Set the initial position of the robot in the map, with x and y in [m] (relative to the origin)
        and theta in [rad]
        """

        x -= self.map.x_range[0]
        y -= self.map.y_range[0]

        self.position = np.array([x, y, theta])

        self.plot_position()

    def plot_position(self):

        x1 = self.position[0]
        x2 = x1 + (2/self.map.resolution) * np.cos(self.position[2]) 

        y1 = self.position[1]
        y2 = y1 + (2/self.map.resolution) * np.sin(self.position[2]) 

        self.p_position[0].set_data(x1, y1)
        self.p_position_rot[0].set_data([x1, x2], [y1, y2])

    def plot_scan(self):

        if self.p_scan is not None:
            self.p_scan.remove()

        scans = list()

        xi = self.position[0]
        yi = self.position[1]
        ti = self.position[2]

        for k in range(self.scan.K):
            trans = self.scan.ranges[k] / self.map.resolution
            angle = self.scan.angle_min + k * self.scan.angle_increment
            xf = xi + trans * np.cos(ti + angle)
            yf = yi + trans * np.sin(ti + angle)
            scans.append([[xi, yi], [xf,yf]])

        lines = LineCollection(list(scans))

        self.p_scan = self.image.add_collection(lines)

        self.p_scan.set_alpha(0.5)

    def plot_particles(self):

        plot_x, plot_y = self.map.pose_to_coord(self.pf.particles[:, 0], self.pf.particles[:, 1])

        offsets = np.column_stack((plot_x, plot_y))

        self.p_particles.set_offsets(offsets)

    def plot_estimate(self):

        x1, y1 = self.map.pose_to_coord(self.estimate[0], self.estimate[1]) # [cells]

        x2 = x1 + 5 * np.cos(self.estimate[2])     # [cells]

        y2 = y1 + 5 * np.sin(self.estimate[2])     # [cells]

        self.p_estimate[0].set_data(x1, y1)
        self.p_estimate_rot[0].set_data([x1, x2], [y1, y2])

    def update_state(self, i):

        result = [self.p_position, self.p_position_rot, self.p_scan, self.p_estimate, self.p_particles]

        if i == 0:
            print("\n\nRunning the algorithm...\n")

        if self.odom is None:
                self.odom = Odom()        
                self.odom.u[0, 2] = self.position[2]
                self.odom.u[1, 2] = self.position[2]

        if self.scan is None:
            angle_min = -2/3 * np.pi  # [rad]
            angle_max = 2/3 * np.pi   # [rad]
            angle_increment = 1/3 * np.pi 
            range_min = 0.19
            range_max = 5.6
            ranges = np.array([3.6055, 3.6055, 5.6, 3.6055, 5.6])
            K = len(ranges)
            self.scan = Scan(angle_min, angle_max, angle_increment, range_min, range_max, ranges, K)

        if i > 2:

            if KIDNAP:
                self.trajectory1(i)
            else:
                # Follow the trajectory without kidnapping
                # Update odometry
                self.trajectory2(i)
                    
            print(f"------------------------------------------------------")
            print(f"Iteration {i}")
            print(str(self.odom))

            self.pf.predict(self.odom, self.map)

            self.pf.update(self.scan, self.map)

            self.pf.resample(self.map)

            self.estimate = self.pf.estimate()

            x, y = self.map.coord_to_pose(self.position[0], self.position[1])

            position = np.array([x, y, self.position[2]])

            print(f"Real Position: {position}\tEstimate: {self.estimate}")

            self.odom.u[0, :] = self.odom.u[1, :]

        self.plot_scan()
        self.plot_position()
        self.plot_estimate()
        self.plot_particles()

        return result
    
    def trajectory1(self, i):
        
        if i < 18:
            
            self.odom.u[1, 1] -= 1              # Move 1 meter in -y direction 
            self.position[1] -= 1 / self.map.resolution  
    
            # Update real scans
            if i == 7:
                self.scan.ranges[4] = 3.6055
        elif i < 27:
            self.odom.u[1, 0] += 1      # Move 1 meter in x direction     
            self.position[0] += 1 / self.map.resolution 
            # Update real scans                 
            if i == 18:
                # Make the turn
                self.odom.u[1, 2] = 0.0  # Change orientation to 0 [rad]
                self.position[2] = 0.0
                self.position[0] = 10
                self.position[1] = 10
                self.scan.ranges[0] = 5.6
                self.scan.ranges[2] = 5.6
                self.scan.ranges[3] = 3.6055
                self.scan.ranges[4] = 5.6
            if i == 21:
                self.scan.ranges[0] = 3.6055
            if i == 23:
                self.scan.ranges[3] = 5.6
                self.scan.ranges[4] = 3.6055
            if i == 25:
                self.scan.ranges[3] = 3.6055
            
        elif i < 39:
            
            self.odom.u[1, 1] += 1              # Move 1 meter in y direction 
            self.position[1] += 1 / self.map.resolution

            if i == 27:
                # Make the turn
                self.odom.u[1, 2] = 0.5 * np.pi  # Change orientation to 0 [rad]
                self.position[2] = 0.5 * np.pi
                self.scan.ranges[0] = 5.6
                self.scan.ranges[1] = 3.6055
                self.scan.ranges[2] = 5.6
                self.scan.ranges[3] = 3.6055
                self.scan.ranges[4] = 5.6
            if i == 28:
                self.scan.ranges[1] = 1.8028
                self.scan.ranges[3] = 2.2361
            if i == 29:
                self.scan.ranges[1] = 1.118
            if i == 30:
                self.scan.ranges[0] = 1.118
                self.scan.ranges[4] = 2.2361
            if i == 36:
                self.scan.ranges[2] = 4.5
            if i > 36:
                self.scan.ranges[2] -= 1      

    def trajectory2(self, i):

        if i < 43:
            
            self.odom.u[1, 1] -= 1              # Move 1 meter in -y direction 
            self.position[1] -= 1 / self.map.resolution  
    
            # Update real scans
            if i == 7:
                self.scan.ranges[4] = 3.6055
            if i == 38:
                self.scan.ranges[3] = 5.6
            if i == 42:
                self.scan.ranges[2] = 5
                self.scan.ranges[4] = 5.6
        elif i < 51:
            self.odom.u[1, 0] += 1      # Move 1 meter in x direction     
            self.position[0] += 1 / self.map.resolution 
            # Update real scans                 
            if i == 43:
                 # Make the turn
                self.odom.u[1, 2] = 0.0  # Change orientation to 0 [rad]
                self.position[2] = 0.0
                self.scan.ranges[0] = 5.6
                self.scan.ranges[2] = 5.6
                self.scan.ranges[3] = 3.6055
                self.scan.ranges[4] = 5.6
            if i == 45:
                self.scan.ranges[0] = 3.6055
            if i == 47:
                self.scan.ranges[3] = 5.6
                self.scan.ranges[4] = 3.6055
            if i == 49:
                self.scan.ranges[3] = 3.6055
        elif i < 63:
            
            self.odom.u[1, 1] += 1              # Move 1 meter in y direction 
            self.position[1] += 1 / self.map.resolution

            if i == 51:
                # Make the turn
                self.odom.u[1, 2] = 0.5 * np.pi  # Change orientation to 0 [rad]
                self.position[2] = 0.5 * np.pi
                self.scan.ranges[0] = 5.6
                self.scan.ranges[1] = 3.6055
                self.scan.ranges[2] = 5.6
                self.scan.ranges[3] = 3.6055
                self.scan.ranges[4] = 5.6
            if i == 52:
                self.scan.ranges[1] = 1.8028
                self.scan.ranges[3] = 2.2361
            if i == 53:
                self.scan.ranges[1] = 1.118
            if i == 54:
                self.scan.ranges[0] = 1.118
                self.scan.ranges[4] = 2.2361
            if i == 60:
                self.scan.ranges[2] = 4.5
            if i > 60:
                self.scan.ranges[2] -= 1
        
class Animation():

    def __init__(self):
        anim = None

    def run(self, ms, frames):
        """
        Run the animation from the ms.update_state function frames times
        """

        self.anim = anmtn.FuncAnimation(plt.gcf(), ms.update_state, frames = frames, interval = 10, repeat = False)
    
    def show(self):
        """
        Show the animation in a new window
        """
        plt.show()

    def save(self, file: str, fps: int):
        """
        Save the animation to the file (.mp4)
        """

        print(f"\nSaving animation to \"{file}\"\n")
        print(f"Animation specifications: \n"
              f"Format = mp4 \n"
              f"FPS    = {fps}")

        FFwriter = anmtn.writers['ffmpeg']
        writer = FFwriter(fps = fps)
        self.anim.save(file, writer)

def main(argv):
    
    ms = MicroSimulator(file = argv[1])

    ms.setInitialPosition(x = 10.0, y = 90.0, theta = -0.5 * np.pi)

    animation = Animation()

    animation.run(ms, frames = 73)

    if (SAVE == False):
        animation.show()
    else:
        animation.save(file = argv[2], fps = 10)

if __name__ == '__main__':
    main(sys.argv[:])

