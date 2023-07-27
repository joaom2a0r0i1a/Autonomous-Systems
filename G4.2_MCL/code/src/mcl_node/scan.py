#!/usr/bin/env python3

class Scan:

    def __init__(self, angle_min, angle_max, angle_increment, range_min, range_max, ranges, K):
        self.angle_min = angle_min              # start angle of the scan [rad]
        self.angle_max = angle_max              # end angle of the scan [rad] 
        self.angle_increment = angle_increment  # angular distance between measurements [rad]
        self.range_min = range_min              # minimum range value [m]
        self.range_max = range_max              # maximum range value [m]
        self.ranges = ranges                    # range data [m] (Note: values < range_min or > range_max should be discarded)
        self.K = K                              # number of observation

    def __str__(self) -> str:
        return (
            f"angle_min       = {self.angle_min}                                                        \n"
            f"angle_max       = {self.angle_max}                                                        \n"
            f"angle_increment = {self.angle_increment}                                                  \n"
            f"range_min       = {self.range_min}                                                        \n"
            f"range_max       = {self.range_max}                                                        \n"
            f"n               = {self.n}                                                                \n"
            f"ranges          =\n{self.ranges}"
        )
    
    def isValid(self, range: float):
        
        if range >= self.range_min and range <= self.range_max:
            return True
        return False

