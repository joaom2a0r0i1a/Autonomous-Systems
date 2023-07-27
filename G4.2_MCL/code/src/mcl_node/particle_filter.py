#!/usr/bin/env python3

import numpy as np

from map import Map
from odom import Odom
from scan import Scan


MM_SD = (0.1, 0.1, 0.1, 0.1)
KT = 1e-20  # Kidnapped Threshold

np.set_printoptions(threshold = np.inf, precision = 5)

class ParticleFilter:

    def __init__(self, N: int):
        self.N = N                          # Number of particles
        self.particles = np.zeros((N, 3))   # N x 3: N rows (1 p/ particle) with 3 columns (for the pose [x y theta])
        self.weights = np.ones(N) / N       # N weights (1 p/ particle) initialized with 1/N for all particles
        self.mean = np.ones(2)              # idx 0: Last Mean, idx 1: Current Mean
        # Results variables
        self.rmse = np.array([])            # RMSE
        self.est_pos = np.zeros(3)          # Our estimate

    def create_uniform_particles(self, x_range, y_range, theta_range, N, map: Map):
        """
        Initialize the N particles uniformly between the specified ranges and in unocuppied cells
        """
        self.particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size = N)
        self.particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size = N)
        self.particles[:, 2] = np.random.uniform(theta_range[0], theta_range[1], size = N)
        
        for n in range(N):
            while not map.isValid(self.particles[n, :]):
                self.particles[n, 0] = np.random.uniform(x_range[0], x_range[1], size = 1)
                self.particles[n, 1] = np.random.uniform(y_range[0], y_range[1], size = 1)

    def create_gaussian_particles(self, mean, std, N, map):
        """
        Initialize the N particles with a gaussian distribution around a specific point and in unocuppied cells
        """
        self.particles[:, 0] = mean[0] + (np.random.randn(N) * std[0])
        self.particles[:, 1] = mean[1] + (np.random.randn(N) * std[1])
        self.particles[:, 2] = mean[2] + (np.random.randn(N) * std[2])
        
        for n in range(N):
            while not map.isValid(self.particles[n, :]):
                self.particles[n, 0] = mean[0] + (np.random.randn(1) * std[0])
                self.particles[n, 1] = mean[1] + (np.random.randn(1) * std[1])

    def predict(self, odom: Odom, map: Map):
        """
        Estimate the poses [x y theta] of the N particles from the odometry readings 'u'.
        """

        u = odom.u

        # Constants for standard deviation
        a_1 = MM_SD[0]
        a_2 = MM_SD[1]
        a_3 = MM_SD[2]
        a_4 = MM_SD[3]

        # Odometry reading from instant 't-1'
        x_bar = u[0,0]
        y_bar = u[0,1]
        theta_bar = u[0,2]

        # Odometry reading from instant 't'
        x_prime_bar = u[1,0]
        y_prime_bar = u[1,1]
        theta_prime_bar = u[1,2]
        
        # Recover relative motion parameters from the odometry readings
        delta_rot_1 = np.arctan2(y_prime_bar-y_bar, x_prime_bar-x_bar) - theta_bar
        delta_trans = np.sqrt(np.square(x_bar-x_prime_bar) + np.square(y_bar-y_prime_bar))
        delta_rot_2 = theta_prime_bar - theta_bar - delta_rot_1

        for n in range(self.N):
            # Generate random sample from relative motion
            delta_rot_1_hat = delta_rot_1 - self.sample_normal_distribution(a_1 * np.absolute(delta_rot_1) + a_2 * delta_trans)
            delta_trans_hat = delta_trans - self.sample_normal_distribution(a_3 * delta_trans + a_4 * (np.absolute(delta_rot_1) + np.absolute(delta_rot_2)))
            delta_rot_2_hat = delta_rot_2 - self.sample_normal_distribution(a_1 * np.absolute(delta_rot_2) + a_2 * delta_trans)

            # Estimate the new pose of particle n
            self.particles[n, 0] = self.particles[n, 0] + delta_trans_hat * np.cos(self.particles[n, 2] + delta_rot_1_hat)
            self.particles[n, 1] = self.particles[n, 1] + delta_trans_hat * np.sin(self.particles[n, 2] + delta_rot_1_hat)
            self.particles[n, 2] = self.particles[n, 2] + delta_rot_1_hat + delta_rot_2_hat

            
            while not map.isValid(self.particles[n, :]):
                self.particles[n, 0] = np.random.uniform(map.x_range[0], map.x_range[1], size = 1)
                self.particles[n, 1] = np.random.uniform(map.y_range[0], map.y_range[1], size = 1)

            # Keep theta between [-PI, PI]
            if self.particles[n, 2] > np.pi:
                self.particles[n, 2] = self.particles[n, 2] - 2 * np.pi
            elif self.particles[n, 2] < -1 * np.pi:
                self.particles[n, 2] = self.particles[n, 2] + 2 * np.pi
            
    def sample_normal_distribution(self, b):
        return 0.5 * np.sum(np.random.uniform(-b,b,12))

    def update(self, scan: Scan, map: Map):
        """
        Update the weights based on particles positions, the measurement and the map
        """

        z_hit = 0.5
        z_rand = 1 - z_hit
        p_rand = 1/scan.range_max

        ranges_true = np.zeros(scan.K)  

        for n in range(self.N):
            
            q = 1

            for k in range(scan.K):
                ranges_true[k] = map.ray_casting(self.particles[n, :], scan.range_max, scan.angle_min + k * scan.angle_increment)
                p_hit = self.p_hit(x = scan.ranges[k], mean = ranges_true[k], sd = np.sqrt(1/(2*np.pi)))
                p = z_hit * p_hit + z_rand * p_rand
                q = p * q                                                               

            self.weights[n] = q

        self.weights += 1e-300

    def p_hit(self, x, mean, sd):
        var = float(sd)**2
        denom = (2*np.pi*var)**.5
        num = np.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom

    def resample(self, map: Map):
        """
        Start by checking if the robot was kidnapped and in that case increment the number of particles and draw 80% with a uniform distribution.
        If Neff < N/2 resample the particles based on their weights, otherwise do nothing.
        """
        self.mean[1] = np.mean(self.weights)    # Update the mean of the non-normalized weights

        self.weights /= sum(self.weights)       # Normalize the weights

        Neff = self.neff()  # Compute Neff   

        if (self.mean < KT).all():
            # The mean of the non-normalized weights is below the Kidnapped Treshold twice in a row so N_new = 1.2 * N 
            # where 80% of N_new are randomly generated
            self.N = int(1.2 * self.N)
            mantain = round(0.2 * self.N)

            new = self.N - mantain

            old_particles_to_mantain = self.particles[0:mantain, :]

            self.particles = np.zeros((new, 3))
            
            self.create_uniform_particles(map.x_range, map.y_range, (- 1 * np.pi, np.pi), new, map)

            particles_total = np.concatenate((self.particles, old_particles_to_mantain), axis = 0)
 
            self.particles = np.zeros((self.N, 3))
            self.weights = np.ones(self.N) / self.N
            
            self.particles = particles_total
        
        elif  Neff < self.N/2:
            # Resample
            cumulative_sum = np.cumsum(self.weights)
            cumulative_sum[-1] = 1. # avoid round-off error
            indexes = np.searchsorted(cumulative_sum, np.random.random(self.N))                                                                                 
         
            self.particles[:] = self.particles[indexes]                                                                              
            self.weights.fill(1.0 / self.N)

        self.mean[0] = self.mean[1]

    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def estimate(self):
        """
        Compute the algorithm estimated position based on the weighted mean of the particles positions
        """
        pos = self.particles[:]
        mean = np.average(pos, weights = self.weights, axis = 0)
        self.est_pos = mean
        return mean

    def compute_rmse(self, amcl_estimate):
        """
        Compute RMSE between our algorithm estimate and the amcl estimate
        """
        x, y = amcl_estimate[0], amcl_estimate[1]
        
        gt_positions = np.array([x, y])
        est_positions = np.array(self.est_pos[:2])
        
        diff = gt_positions - est_positions
        rmse = np.sqrt(np.sum(diff**2))
        self.rmse = np.append(self.rmse, rmse)

    def print_particles(self):
        for n in range(self.N):
            print(f"Particle {n:2} {self.particles[n, :]}")






