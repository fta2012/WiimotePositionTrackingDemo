A demo of how to do simple 6DOF position tracking using a wiimote. Requires 4 IR LEDs as tracking markers. [Blog](http://franklinta.com/2014/09/30/6dof-positional-tracking-with-the-wiimote/) has details on how to make one.

Uses [wiiuse](https://github.com/rpavlik/wiiuse) for connecting to the wiimote and getting raw IR values. Uses OpenCV's [solvepnp](http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#solvepnp) for solving for camera pose. Drawing was based off of wiiuse's example which requires SDL 1.2 and OpenGL.

### Build and run:
    git submodule init
    git submodule update
    mkdir build
    cd build
    cmake ..
    make
    ./demo

Press the sync button on the back of a wiimote to connect while it is searching for devices.

Controls:

    A - switch between world frame and camera frame
    B - draw the current path
    Home - clear the drawn path
    UP/DOWN - change the rendered camera size
    LEFT/RIGHT - rotate the mapping of the leds (press this if your world y-axis is not pointing up or your camera y-axis is not pointing down)
    PLUS/MINUS - change IR sensitivity
    ONE - toggle whether to draw the wiimote
    TWO - use the last few frames to print out a calibrated camera matrix
