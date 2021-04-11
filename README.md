# OpenCV INSTALL
Because SIRF and SURF are no longer available in opencv > 3.4.2.16, so a proper version should be download or you can easily
get an error running the code like this :  'module' object has no attribute 'xfeatures2d' 

### check opencv version
First open a terminal
```
python
import cv2
cv2.__version__
quit()
```
My version of cv2 is 3.4.2.
```
pip2 install opencv-python==3.4.2.16
pip2 install opencv-contrib-python==3.4.2.16
```
💫One nonnegligible point is python2 and python3 can have different opencv version.
It means if you run "pip install", the package may be installed to python3 while our code is based on python2.
Don't forget this.

# Installing
After the basic installation on "detection" branch is done, 
```
$ git clone -b 6D-percept https://github.com/tjr16/dd2419_detector_baseline.git
```

# Running

### Same as before
```
$ rosrun perception yolo_detector.py
```
