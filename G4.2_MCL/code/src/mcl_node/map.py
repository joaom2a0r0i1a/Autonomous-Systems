#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class Map:
    """
    Map object to use in the MCL algorithm. Stores the necessary variables and has methods for the desired operations
    """
    def __init__(self, data, width, height, resolution, x_range, y_range):
        # Variables for the algorithm
        self.data = data                # HEIGHT x WIDTH matrix representation
        self.width = width              # [cell]
        self.height = height            # [cell]
        self.resolution = resolution    # [m/cell]
        self.x_range = x_range          # [m]
        self.y_range = y_range          # [m]

        # Plotting
        self.scatter_p = plt.scatter([0], [0], alpha = 0.5, color = 'g', s = 2, label = "Particles")                        # Particles
        self.scatter_e = plt.scatter([0], [0], alpha = 1, marker = '+', color = 'r', s = 5, label = "Algorithm Estimate")    # Estimate
        self.scatter_ae = plt.scatter([0], [0], alpha = 1, marker = '+', color = 'b', s = 5, label = "AMCL Estimate")        # Estimate

    def __str__(self):
        return (
            f"Map specifications: \n"
            f"Width      = {self.width} \n"
            f"Height     = {self.height} \n"
            f"Resolution = {self.resolution:.3f} \n"
            f"Range x    = ({self.x_range[0]:.3f}, {self.x_range[1]:.3f}) \n"
            f"Range y    = ({self.y_range[0]:.3f}, {self.y_range[1]:.3f}) \n"
        )    

    def isValid(self, particle):
        """
        Checks if a particle pose [in meters] is valid on the current map
        """
        x, y = self.pose_to_coord(particle[0], particle[1])

        if x < 0 or x > self.width or y < 0 or y > self.height:
            return False

        if self.data[int(y), int(x)] != 0:
            return False
        
        return True

    def ray_casting(self, pose, range, angle):
        """
        Given a pose, a max laser range and an angle of the beam draws the line between (x1, y1) and (x2, y2) and computes the closest interception with a wall, or
        if there is not one return the maximum range of the sensor
        """
        x1, y1 = self.pose_to_coord(pose[0], pose[1])

        x1 = round(x1)
        y1 = round(y1)

        (x2, y2) = (round(x1 + (range / self.resolution) * np.cos(angle + pose[2])), round(y1 + (range / self.resolution) * np.sin(angle + pose[2])))

        line = list(self.bresenham(x1, y1, x2, y2))

        for x,y in line:
            # x, y are coordinates [cells]
            pose_x, pose_y = self.coord_to_pose(x, y)    # [meters]
            if not self.isValid(np.array([pose_x, pose_y])):
                dist = np.linalg.norm(np.array([pose_x, pose_y]) - np.array([pose[0], pose[1]]))
                if dist < range:
                    return dist
                else:
                    break

        return range

    def bresenham(self, x0, y0, x1, y1):
        """
        Draw the line between two points
        """
        dx = x1 - x0
        dy = y1 - y0

        xsign = 1 if dx > 0 else -1
        ysign = 1 if dy > 0 else -1

        dx = abs(dx)
        dy = abs(dy)

        if dx > dy:
            xx, xy, yx, yy = xsign, 0, 0, ysign
        else:
            dx, dy = dy, dx
            xx, xy, yx, yy = 0, ysign, xsign, 0

        D = 2*dy - dx
        y = 0

        for x in range(dx + 1):
            yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
            if D >= 0:
                y += 1
                D -= 2*dx
            D += 2*dy

    def plot_particles(self, particles):
        """
        Function to plot the particles, converts the pose to the correct cell in the map
        """
        self.scatter_p.remove()
        
        plot_x, plot_y = self.pose_to_coord(particles[:, 0], particles[:, 1])
       
        self.scatter_p = plt.scatter(plot_x, plot_y, alpha = 0.5, color = 'g', s = 2, label = "Particles")

        plt.draw()
        plt.pause(0.00000000001)

    def plot_estimate(self, estimate):
        """
        Function to plot the estimate, converts the pose to the correct cell in the map
        """

        self.scatter_e.remove()

        plot_x, plot_y = self.pose_to_coord(estimate[0], estimate[1])

        self.scatter_e = plt.scatter(plot_x, plot_y, alpha = 1, marker = '+', color = 'r', s = 5, label = "Algorithm Estimate")

        plt.draw()
        plt.pause(0.00000000001)

    def plot_amcl_estimate(self, estimate):
        """
        Function to plot the amcl_estimate, converts the pose to the correct cell in the map
        """
        self.scatter_ae.remove()

        plot_x, plot_y = self.pose_to_coord(estimate[0], estimate[1])

        self.scatter_ae = plt.scatter(plot_x, plot_y, alpha = 1, marker = '+', color = 'b', s = 5, label = "AMCL Estimate")

        plt.draw()
        plt.pause(0.00000000001)

    def pose_to_coord(self, pose_x, pose_y):
        """
        Converts pose [meters] to 2D coordinates on the map [cells]
        """
        x = np.divide(np.subtract(pose_x, self.x_range[0]), self.resolution)
        y = np.divide(np.subtract(pose_y, self.y_range[0]), self.resolution)
        return x, y

    def coord_to_pose(self, x, y):
        """
        Converts 2D coordinates on the map to [cells] to pose [meters]
        """
        pose_x = np.add(np.multiply(x, self.resolution), self.x_range[0]) 
        pose_y = np.add(np.multiply(y, self.resolution), self.y_range[0]) 
        return pose_x, pose_y

    
