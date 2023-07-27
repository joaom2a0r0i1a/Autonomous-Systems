#!/usr/bin/env python3

import numpy as np

class Odom:
    """
    Odometry class to use in MCL algorithm
    """
    def __init__(self):
        self.u = np.zeros((2,3))    # 2 x 3 matrix where row 0 is the odometry reading at instant 't-1' and row 1 is odometry reading at instant 't'

    def __str__(self):
        return (f"Odometry reading @ t-1: {self.u[0, :]}                                                                 \n"
                f"Odometry reading @ t:\t{self.u[1, :]}                                                                    "
        )   