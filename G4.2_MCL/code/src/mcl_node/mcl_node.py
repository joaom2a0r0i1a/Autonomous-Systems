#!/usr/bin/env python3

# command to run: roslaunch mcl_node mcl.launch
# To change the map to load and the bag to play edit the mcl.launch file.
import numpy as np
import matplotlib.pyplot as plt

import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion

from map import Map
from odom import Odom
from scan import Scan
from particle_filter import ParticleFilter

np.set_printoptions(threshold = np.inf, precision = 5)

# Number of particles
N = 400

i = 1

class MCLNode:

    def __init__(self):

        # Initialize some necessary variables here
        self.node_frequency = None
        self.sub_map_topic = None
        self.sub_pose_topic = None
        self.sub_scan_topic = None
        self.sub_amcl_pose_topic = None
       
        # Create the particle filter object
        self.pf = ParticleFilter(N)

        # Variables for the Map, Odom and Scan objects:
        self.map = None             # In the 'callback_map_topic' initialize as a Map object with the important variables to store for the algorithm
        self.odom = None            # In the 'callback_pose_topic' initialize as an Odom object with the important variables to store for the algorithm
        self.scan = None            # In the 'callback_scan_topic' initialize as a Scan object with the important variables to store for the algorithm
        self.amcl_estimate = None   # In the 'callback_amcl_pose_topic' store the amcl_estimate
        
        self.time = np.array([])

        self.map_msg_received = False   # Flag to indicate if a a map message was received
        self.odom_msg_received = False  # Flag to indicate if a a odom message was received
        self.scan_msg_received = False  # Flag to indicate if a a scan message was received

        # Initialize the ROS node
        rospy.init_node('mcl_node')
        rospy.loginfo_once('MCL Node has started')

        # Load parameters from the parameter server
        self.load_parameters()

        # Initialize the subscribers
        self.initialize_subscribers()
        
        # Initialize the timer with the corresponding interruption to work at a constant rate
        self.initialize_timer()

    def load_parameters(self):
        """
        Load the parameters from the configuration server (ROS)
        """

        # Node frequency of operation
        self.node_frequency = rospy.get_param('node_frequency', 1)
        rospy.loginfo('Node Frequency: %s', self.node_frequency)

    def initialize_subscribers(self):
        """
        Initialize the subscribers to the topics '/map', '/pose', '/scan' and '/amcl_pos'
        """

        # Subscribe to the topics '/map', '/pose' and '/scan'
        self.sub_map_topic = rospy.Subscriber('/map', OccupancyGrid, self.callback_map_topic)
        self.sub_pose_topic = rospy.Subscriber('/pose', Odometry, self.callback_pose_topic)
        self.sub_scan_topic = rospy.Subscriber('/scan', LaserScan, self.callback_scan_topic)
        self.sub_amcl_pose_topic = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.callback_amcl_pose_topic)
        
        rospy.loginfo('Subscribed to: %s', self.sub_map_topic.name)
        rospy.loginfo('Subscribed to: %s', self.sub_pose_topic.name)
        rospy.loginfo('Subscribed to: %s', self.sub_scan_topic.name)
        rospy.loginfo('Subscribed to: %s', self.sub_amcl_pose_topic.name)

    def initialize_timer(self):
        """
        Here we create a timer to trigger the callback function at a fixed rate.
        """

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.node_frequency), self.timer_callback)
        self.h_timerActivate = True

    def timer_callback(self, timer):
       
        global i

        if self.map_msg_received and self.odom_msg_received  and self.scan_msg_received:

            print(f"-----------------------------------------------------------------------------------------")
            print(f"Iteration {i}")
            i += 1
            print(str(self.odom))

            if self.odom.u[0, 0] == 0.0:
                # Discard initial jump from initialization
                self.odom.u[0, :] = self.odom.u[1, :]
                return

            # Predict the poses of the particles in the set from the motion model
            self.pf.predict(self.odom, self.map)

            self.odom.u[0, :] = self.odom.u[1, :]

            # Update the weights of the particles in the set from the observation model
            self.pf.update(self.scan, self.map)

            # Resamples the new set so all the particles have the same weight and are 
            # drawn according to the weight before the resample
            self.pf.resample(self.map)

            mu = self.pf.estimate()

            print(f"Our Estimate {mu}, AMCL Estimate {self.amcl_estimate}")

            self.map.plot_particles(self.pf.particles)
            self.map.plot_estimate(mu)
            self.map.plot_amcl_estimate(self.amcl_estimate)
            
            # Results information (change for each test!)
            self.pf.compute_rmse(self.amcl_estimate)
            
            now = rospy.get_rostime()                        
            self.time = np.append(self.time, now.secs - 1686825086)

            if len(self.time) in [30,31,32,33,34,35,36,37,38,39]:
                print(self.time)  
                print(self.pf.rmse)                              
            
    def callback_map_topic(self, msg: OccupancyGrid):
        """
        Callback function for the subscriber of the topic '/map'. This function is called whenever
        a message is received by the subscriber 'self.sub_map_topic'. The message received is 
        processed and stored in 'self.map' (a Map object) to be used later in the 'timer_callback'.
        Refer to http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/OccupancyGrid.html to see the 
        structure of the message.
        """

        self.map_msg_received = True

        # Local variables
        data = np.asarray(msg.data)
        width = msg.info.width              # Number of cells along the x axis [cell]
        height = msg.info.height            # Number of cells along the y axis [cell]
        resolution = msg.info.resolution    # [m/cell]

        # Rearrange so data is HEIGHT x WIDTH  [cells] matrix, easier for coordinate use
        data.shape = height, width
        data[data == -1] = 20

        # Obtain the range in [meters] for the poses
        x_min = msg.info.origin.position.x              # cell(0,0) x value in [m]
        y_min = msg.info.origin.position.y              # cell(0,0) y value in [m]
        x_max = x_min + (width - 1) * resolution        # Last cell x value in [m]
        y_max = y_min + (height - 1) * resolution       # Last cell y value in [m]

        x_range = (x_min, x_max)    # [m]
        y_range = (y_min, y_max)    # [m]

        # Create the Map object
        self.map = Map(data, width, height, resolution, x_range, y_range)

        print(f"Loading map...\n")
        print(str(self.map))

        # Initialize the particles
        print(f"Initializing {N} particles\n")
        
        # For Gaussian Distribution:
        """
        pose_x, pose_y = self.map.coord_to_pose(x = 320, y = 310)
        mean = np.array([pose_x, pose_y, -0.5 * np.pi])
        std = np.array([1, 1, 0.25 * np.pi])
        self.pf.create_gaussian_particles(mean = mean, std = std, N = N, map = self.map)
        """
        # For Uniform Distribution
        self.pf.create_uniform_particles(x_range, y_range, (-1 * np.pi, 1 * np.pi), N, self.map)
        self.map.plot_particles(self.pf.particles) 

        # Plot the map data in a figure 
        plt.imshow(data, origin = 'lower', cmap = 'Greys', interpolation = 'bilinear')  
        plt.legend()
                      
    def callback_pose_topic(self, msg: Odometry):
        """
        Callback function for the subscriber of the topic '/pose'. This function is called whenever
        a message is received by the subscriber 'self.sub_pose_topic'. The message received is 
        processed and stored in 'self.odom' to be used later in the 'timer_callback'.
        Refer to http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html to see the 
        structure of the message.
        """

        self.odom_msg_received = True

        if self.odom is None:
            self.odom = Odom()

        # Get the odometry reading from the message and convert to [x y theta]
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

        # Update odometry reading at instant 't'
        self.odom.u[1, :] = np.array([position.x, position.y, yaw])
           
    def callback_scan_topic(self, msg: LaserScan):
        """
        Callback function for the subscriber of the topic '/scan'. This function is called
        whenever a message is received by the 'self.sub_scan_topic'. The message received is 
        processed and stored in 'self.scan' to be used later in the 'timer_callback'.
        Refer to http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html to see the 
        structure of the message.
        """

        self.scan_msg_received = True

        angle_min = msg.angle_min               # [rad]
        angle_max = msg.angle_max               # [rad]
        angle_increment = msg.angle_increment   # [rad]
        range_min = msg.range_min               # [m]
        range_max = msg.range_max               # [m]
        ranges = np.asarray(msg.ranges)         # [m]
        ranges[np.isnan(ranges)]  = range_max
        K = len(ranges)                         # Number of observations in the LaserScan

        desired_angle_increment = 10 * np.pi / 180   # [rad]

        jump = round(desired_angle_increment/angle_increment)

        new_K = round(K/jump)
        new_ranges = np.zeros(new_K)
        
        j = 0
        for i in range(K):
            if i%jump == 0:
                new_ranges[j] = ranges[i]
                j += 1

        # Create a Scan object with the necessary variables
        self.scan = Scan(angle_min, angle_max, desired_angle_increment, range_min, range_max, new_ranges, new_K)

    def callback_amcl_pose_topic(self, msg: PoseWithCovarianceStamped):
        """
        Callback function for the subscriber of the topic '/amcl_pose'. This function is called
        whenever a message is received by the 'self.sub_amcl_pose_topic'. The message received is 
        processed and stored in 'self.amcl_estimate to be used later in the 'timer_callback'.
        Refer to http://docs.ros.org/en/api/geometry_msgs/html/msg/PoseWithCovarianceStamped.html to see the 
        structure of the message.
        """
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

        self.amcl_estimate = np.array([position.x, position.y, yaw])

def main():

    # Create a figure
    plt.figure()

    # Create an instance of the MCLNode class
    mcl_node = MCLNode()

    # Show the figure at each iteration
    plt.show()

    # Spin to keep the script for exiting
    rospy.spin()
    
if __name__ == '__main__':
    main()

