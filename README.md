# dd2419_detector_baseline
Traffic signs detection based on a simple detector baseline for DD2419 course

# Prerequisites

- Ubuntu 18.04
- ROS Melodic
- OpenCV

# Installing
### Install ros package
```
$ catkin_create_pkg perception rospy geometry_msgs tf2_ros tf std_msgs crazyflie_driver
$ source ../devel/setup.$(basename $SHELL)
$ mkdir perception
$ git clone -b detection https://github.com/tjr16/dd2419_detector_baseline.git
```
Extract all the files from `dd2419_detector_baseline` to `perception` and then build 
```
$ catkin build
```

### Install opencv <for feature detection, not implemented yet>
```
$ To be filled
```
# Running
### *** Tips: to increase functionality of CPU ***
You can type the following line to check current CPU Running frequency:
```
grep MHz /proc/cpuinfo
```
If it's largely below the standard frequency the product is said to be, probably check if you PC is on powersave mode:
https://askubuntu.com/questions/929884/how-to-set-performance-instead-of-powersave-as-default
```
sudo systemctl restart cpufrequtils
```
Above line is just for convenience for me.
### Get ros node started
```
$ rosrun perception yolo_detector.py
```
### Create a tf from map to odom (if you already have one, skip this step:)
```
$ rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 map cf1/odom
```
💫 most things should be visualized in Rviz! Check image sending from sign/detection & the tf xD!

### Check message information
```
$ rosmsg info perception/Sign
$ rosmsg info perception/SignArray
```
### Check rostopic 
```
$ rostopic echo sign/detected
```
The echoing information should be of type SignArray which refers to an array of the signs it is detected currently, empty if no detection found

### Nice reminder
- The confidence in the config.py file decides the tolerance for detection. Higher confidence would easily reject the detection and vise versa. If you cannot detection anything while there should be, try lower this value. 
- Distance > 0.5m makes classification somehow unstable :(
