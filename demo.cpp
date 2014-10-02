#include <algorithm>
#include <chrono>
#include <thread>

#include <SDL.h>
#include <SDL_error.h>
#include <SDL_events.h>
#include <SDL_video.h>

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>

#include <opencv2/opencv.hpp>

#include <wiiuse.h>

using namespace std;

// Dimensions of the image taken by the IR camera
int image_width = 1024;
int image_height = 768;

// Location of the infrared LEDs in world frame.
// This tracker is in the shape of a square.
vector<cv::Point3f> object_points = {
    {1, 1, 0},
    {1, 3, 0},
    {3, 3, 0},
    {3, 1, 0},
};

// The camera's position in world frame while 'B' was held
vector<cv::Mat> camera_path;

// Queue recording the last few images where all 4 points were visible.
deque<vector<cv::Point2f>> image_points_queue;

// Whether we should be drawing using the camera or world coordinate frame
enum render_mode_t {
    CAMERA_FRAME = 1,
    WORLD_FRAME
};
enum render_mode_t render_mode = CAMERA_FRAME;

// How large to draw the camera
float camera_scale = 3;

// Whether to draw the wiimote
bool draw_wiimote = true;


void handle_event(struct wiimote_t* wiimote) {
    if (IS_JUST_PRESSED(wiimote, WIIMOTE_BUTTON_LEFT)) {
        rotate(object_points.begin(), object_points.end() - 1, object_points.end());
    }
    if (IS_JUST_PRESSED(wiimote, WIIMOTE_BUTTON_RIGHT)) {
        rotate(object_points.begin(), object_points.begin() + 1, object_points.end());
    }

    if (IS_JUST_PRESSED(wiimote, WIIMOTE_BUTTON_UP)) {
        camera_scale += 1;
    }
    if (IS_JUST_PRESSED(wiimote, WIIMOTE_BUTTON_DOWN)) {
        camera_scale -= 1;
    }

    if (IS_JUST_PRESSED(wiimote, WIIMOTE_BUTTON_A)) {
        render_mode = (render_mode == CAMERA_FRAME) ? WORLD_FRAME : CAMERA_FRAME;
    }

    if (IS_JUST_PRESSED(wiimote, WIIMOTE_BUTTON_HOME)) {
        camera_path.clear();
    }

    if (IS_JUST_PRESSED(wiimote, WIIMOTE_BUTTON_PLUS)) {
        int level;
        WIIUSE_GET_IR_SENSITIVITY(wiimote, &level);
        wiiuse_set_ir_sensitivity(wiimote, level + 1);
    }
    if (IS_JUST_PRESSED(wiimote, WIIMOTE_BUTTON_MINUS)) {
        int level;
        WIIUSE_GET_IR_SENSITIVITY(wiimote, &level);
        wiiuse_set_ir_sensitivity(wiimote, level - 1);
    }

    if (IS_JUST_PRESSED(wiimote, WIIMOTE_BUTTON_ONE)) {
        draw_wiimote = !draw_wiimote;
    }

    if (IS_JUST_PRESSED(wiimote, WIIMOTE_BUTTON_TWO)) {
        if (!image_points_queue.empty()) {
            cout << "Calibrating... " << endl;
            vector<vector<cv::Point3f>> objectPointsVector(image_points_queue.size(), object_points);
            cv::Mat camera_matrix, dist_coeffs;
            vector<cv::Mat> rvecs, tvecs;
            calibrateCamera(objectPointsVector,
                            vector<vector<cv::Point2f>>(image_points_queue.begin(), image_points_queue.end()),
                            cvSize(image_width, image_height),
                            camera_matrix,
                            dist_coeffs,
                            rvecs,
                            tvecs);
            cout << "cameraMatrix " << camera_matrix << endl;
            cout << "distCoeffs " << dist_coeffs << endl;
        }
    }
}

vector<cv::Point2f> get_image_points(wiimote* wiimote) {
    vector<cv::Point2f> image_points;

    for (const auto & dot : wiimote->ir.dot) {
        // Only keep visible points
        if (dot.visible) {
            // Flip so that (0, 0) corresponds with the top left corner of the image
            image_points.emplace_back(image_width - 1 - dot.rx, image_height - 1 - dot.ry);
        }
    }

    // If all 4 points are visible, canonicalize the ordering
    if (image_points.size() == object_points.size()) {
        // Make sure it is in counterclockwise order so it can match up with object points
        convexHull(image_points, image_points);
        if (image_points.size() == object_points.size()) {
            // If there is a previous image, try to rotate the points until they match up
            if (!image_points_queue.empty()) {
                auto & previous_image = image_points_queue.back();
                float min_dist = FLT_MAX;
                int min_offset = 0;
                for (int offset = 0; offset < image_points.size(); offset++) {
                    float dist = 0;
                    for (int i = 0; i < image_points.size(); i++) {
                        cv::Point2f diff = previous_image[i] - image_points[(i + offset) % image_points.size()];
                        dist += norm(diff);
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_offset = offset;
                    }
                }
                rotate(image_points.begin(), image_points.begin() + min_offset, image_points.end());
            }

            // Save the canonicalized image for this frame
            image_points_queue.push_back(image_points);
            // Limit the number of frames to keep
            while (image_points_queue.size() > 15) {
                image_points_queue.pop_front();
            }
        }
    }

    return image_points;
}

void display(wiimote* wiimote) {
    vector<cv::Point2f> image_points = get_image_points(wiimote);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (render_mode == CAMERA_FRAME) {
        // Draw the LEDs in 2D
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0, image_width, image_height, 0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glPointSize(5.0);
        glBegin(GL_POINTS);
        for (int i = 0; i < image_points.size(); i++) {
            glColor3f(i == 0 || i == 3, i == 1 || i == 3, i == 2 || i == 3);
            glVertex2f(image_points[i].x, image_points[i].y);
        }
        glEnd();
    }

    if (image_points.size() < object_points.size()) {
        // Not all 4 points are visible so we can't solve, just return
        if (render_mode == CAMERA_FRAME) {
            SDL_GL_SwapBuffers();
        }
        return;
    }

    // Camera intrinsic parameters. These weren't obtained from calibration but works well enough.
    double fx = 1700;
    double fy = 1700;
    double cx = image_width / 2;
    double cy = image_height / 2;
    cv::Mat intrinsic = (cv::Mat_<double>(3, 3) <<
        fx,  0, cx,
         0, fy, cy,
         0,  0,  1
    );

    // Solve for camera extrinsic parameters.
    // This gives us the rotation and translation of the world frame from the camera frame.
    cv::Mat rvec, tvec;
    solvePnP(object_points, image_points, intrinsic, cv::noArray(), rvec, tvec);
    cv::Mat R;
    Rodrigues(rvec, R);
    cv::Mat extrinsic = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(extrinsic.rowRange(0, 3).colRange(0, 3));
    tvec.copyTo(extrinsic.rowRange(0, 3).col(3));

    // Find the inverse of the extrinsic matrix (should be the same as just calling extrinsic.inv())
    cv::Mat extrinsic_inv_R = R.t(); // inverse of a rotational matrix is its transpose
    cv::Mat extrinsic_inv_tvec = -extrinsic_inv_R * tvec;
    cv::Mat extrinsic_inv = cv::Mat::eye(4, 4, CV_64F);
    extrinsic_inv_R.copyTo(extrinsic_inv.rowRange(0, 3).colRange(0, 3));
    extrinsic_inv_tvec.copyTo(extrinsic_inv.rowRange(0, 3).col(3));

    // Find the inverse of the intrinsic matrix
    cv::Mat intrinsic_inv = (cv::Mat_<double>(4, 4) <<
        1 / fx,      0, -cx / fx,  0,
             0, 1 / fy, -cy / fy,  0,
             0,      0,        1,  0,
             0,      0,        0,  1
    );

    // Record the position of the camera, which is (extrinsic_inv * [0, 0, 0, 1])
    if (IS_PRESSED(wiimote, WIIMOTE_BUTTON_B)) {
        camera_path.push_back(extrinsic_inv_tvec);
    }

    if (render_mode == CAMERA_FRAME) {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        // Since our intrinsic matrix has center in the middle of image we can use gluPerspective with the correct fov and aspect.
        gluPerspective(2 * atan(cy / fy) * 180 / M_PI, fy * image_width / (fx * image_height), 0.1, 100.0);

        // The front of the camera in computer vision is the positive z-axis but is the negative z-axis in opengl
        // Rotate the z axis around
        GLfloat cv_to_gl[16] = {
            1,  0,  0,  0,
            0, -1,  0,  0,
            0,  0, -1,  0,
            0,  0,  0,  1,
        };
        glMultMatrixf(cv_to_gl);

        // Apply the extrinsic matrix in column major order.
        // We can now draw stuff in world coordinates and to show what the IR camera would see.
        glMultMatrixd(cv::Mat(extrinsic.t()).ptr<double>(0));
    } else {
        // Some arbitrary fixed viewing direction of the world frame
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(60.0f, (float)image_width / image_height, 0.1f, 100.0f);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(10, 10, 30, 1.5, 1.5, 0, 0, 1, 0);
    }

    /* Draw in world coordinates */

    // Draw world frame axes
    glBegin(GL_LINES);
    glColor3f(1.0, 0.0, 0.0); glVertex3f(0, 0, 0); glVertex3f(1000, 0, 0);
    glColor3f(0.0, 1.0, 0.0); glVertex3f(0, 0, 0); glVertex3f(0, 1000, 0);
    glColor3f(0.0, 0.0, 1.0); glVertex3f(0, 0, 0); glVertex3f(0, 0, 1000);
    glEnd();

    // Draw the square representing the LEDs
    glBegin(GL_LINE_LOOP);
    glColor3f(1.0, 1.0, 1.0);
    for (const auto & object_point : object_points) {
        glVertex3f(object_point.x, object_point.y, 0);
    }
    glEnd();

    // Draw lines from the leds to where they are projected on the camera
    vector<cv::Point2f> projected_points;
    projectPoints(object_points, rvec, tvec, intrinsic, cv::noArray(), projected_points);
    glColor3f(0.0, 0.0, 1.0);
    glBegin(GL_LINES);
    for (int i = 0; i < object_points.size(); i++) {
        glVertex3f(object_points[i].x, object_points[i].y, 0);
        cv::Mat p = (cv::Mat_<double>(4, 1) << projected_points[i].x * camera_scale, projected_points[i].y * camera_scale, camera_scale, 1);
        p = extrinsic_inv * intrinsic_inv * p;
        assert(p.at<double>(3) == 1);
        glVertex3f(p.at<double>(0), p.at<double>(1), p.at<double>(2));
    }
    glEnd();

    // Draw the camera path
    glBegin(GL_LINES);
    glColor3f(1.0, 1.0, 1.0);
    for (int i = 1; i < camera_path.size(); i++) {
        if (cv::norm(camera_path[i], camera_path[i - 1]) > .5) // Skip consecutive points that are too far apart
            continue;
        glVertex3f(camera_path[i - 1].at<double>(0), camera_path[i - 1].at<double>(1), camera_path[i - 1].at<double>(2));
        glVertex3f(camera_path[i].at<double>(0), camera_path[i].at<double>(1), camera_path[i].at<double>(2));
    }
    glEnd();

    /* Draw in camera frame */
    glMultMatrixd(cv::Mat(extrinsic_inv.t()).ptr<double>(0));

    // Draw camera frame axes
    glBegin(GL_LINES);
    glColor3f(1.0, 0.0, 0.0); glVertex3f(0.0, 0.0, 0.0); glVertex3f(3.0, 0.0, 0.0);
    glColor3f(0.0, 1.0, 0.0); glVertex3f(0.0, 0.0, 0.0); glVertex3f(0.0, 3.0, 0.0);
    glEnd();

    // Draw the wiimote
    if (draw_wiimote) {
        float wiimote_width = 1.43;
        float wiimote_height = 1.21;
        float wiimote_length = 5.8;
        glColor3f(1, 1, 1);
        glBegin(GL_LINE_LOOP);
        glVertex3f( wiimote_width / 2,  wiimote_height / 2, 0);
        glVertex3f( wiimote_width / 2, -wiimote_height / 2, 0);
        glVertex3f(-wiimote_width / 2, -wiimote_height / 2, 0);
        glVertex3f(-wiimote_width / 2,  wiimote_height / 2, 0);
        glEnd();
        glBegin(GL_LINES);
        glVertex3f( wiimote_width / 2,  wiimote_height / 2, 0); glVertex3f( wiimote_width / 2,  wiimote_height / 2, -wiimote_length);
        glVertex3f( wiimote_width / 2, -wiimote_height / 2, 0); glVertex3f( wiimote_width / 2, -wiimote_height / 2, -wiimote_length);
        glVertex3f(-wiimote_width / 2, -wiimote_height / 2, 0); glVertex3f(-wiimote_width / 2, -wiimote_height / 2, -wiimote_length);
        glVertex3f(-wiimote_width / 2,  wiimote_height / 2, 0); glVertex3f(-wiimote_width / 2,  wiimote_height / 2, -wiimote_length);
        glEnd();
        glBegin(GL_LINE_LOOP);
        glVertex3f( wiimote_width / 2,  wiimote_height / 2, -wiimote_length);
        glVertex3f( wiimote_width / 2, -wiimote_height / 2, -wiimote_length);
        glVertex3f(-wiimote_width / 2, -wiimote_height / 2, -wiimote_length);
        glVertex3f(-wiimote_width / 2,  wiimote_height / 2, -wiimote_length);
        glEnd();
    }

    /* Draw on the image plane of the camera */
    glMultMatrixd(cv::Mat(intrinsic_inv.t()).ptr<double>(0));

    vector<cv::Point3f> image_plane = {
        {0,                                                    0, camera_scale},
        {image_width * camera_scale,                           0, camera_scale},
        {image_width * camera_scale, image_height * camera_scale, camera_scale},
        {0,                          image_height * camera_scale, camera_scale},
    };

    // Draw the boundaries for the image plane
    glBegin(GL_LINE_LOOP);
    glColor3f(0.5, 0.5, 0.5);
    for (const auto & corner : image_plane) {
        glVertex3f(corner.x, corner.y, corner.z);
    }
    glEnd();

    // Draw the lines connecting the plane to the pinhole of the camera
    glBegin(GL_LINES);
    glColor3f(0.5, 0.5, 0.5);
    for (const auto & corner : image_plane) {
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(corner.x, corner.y, corner.z);
    }
    glEnd();

    // Draw the image points on the image plane
    glColor3f(1.0, 0.0, 0.0);
    glBegin(GL_LINE_LOOP);
    for (const auto & p : image_points) {
        glVertex3f(p.x * camera_scale, p.y * camera_scale, camera_scale);
    }
    glEnd();

    // Draw the projected object points on the image plane
    // Whenever an exact solution is found the actual image points (drawn in red) should be covered by this
    glColor3f(1.0, 1.0, 1.0);
    glBegin(GL_LINE_LOOP);
    for (const auto & p : projected_points) {
        glVertex3f(p.x * camera_scale, p.y * camera_scale, camera_scale);
    }
    glEnd();

    SDL_GL_SwapBuffers();
}

int main(int argc, char** argv) {
    wiimote** wiimotes = wiiuse_init(1);
    int found = wiiuse_find(wiimotes, 1, 5);
    if (!found) {
        printf("Failed to find any wiimote.\n");
        return 0;
    }
    int connected = wiiuse_connect(wiimotes, 1);
    if (connected) {
        printf("Connected to %i wiimotes (of %i found).\n", connected, found);
    } else {
        printf("Failed to connect to any wiimote.\n");
        return 0;
    }
    wiiuse_rumble(wiimotes[0], 1);
    this_thread::sleep_for(chrono::milliseconds(200));
    wiiuse_rumble(wiimotes[0], 0);
    wiiuse_set_leds(wiimotes[0], WIIMOTE_LED_1);
    wiiuse_set_ir(wiimotes[0], 1);

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("Failed to initialize SDL: %s\n", SDL_GetError());
        return 0;
    }
    SDL_WM_SetCaption("Wiimote camera pose estimation", "Wiimote camera pose estimation");
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
    SDL_SetVideoMode(image_width, image_height, 16, SDL_OPENGL);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearColor(0, 0, 0, 0);
    glViewport(0, 0, image_width, image_height);

    chrono::high_resolution_clock::time_point last_render;
    chrono::high_resolution_clock::time_point last_report;
    int fps = 0;

    display(wiimotes[0]);
    while (1) {
        SDL_Event event;
        if (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    SDL_Quit();
                    wiiuse_cleanup(wiimotes, 1);
                    return 0;
                default:
                    break;
            }
        }

        if (wiiuse_poll(wiimotes, 1)) {
            switch (wiimotes[0]->event) {
                case WIIUSE_EVENT:
                    handle_event(wiimotes[0]);
                    break;
                default:
                    break;
            }
        }

        auto now = chrono::high_resolution_clock::now();
        if (now - last_report >= std::chrono::seconds(1)) {
            printf("fps: %d\n", fps);
            fps = 0;
            last_report = now;
        }
        if (now - last_render >= std::chrono::milliseconds(1000) / 60) {
            display(wiimotes[0]);
            fps++;
            last_render = now;
        }
    }
}
