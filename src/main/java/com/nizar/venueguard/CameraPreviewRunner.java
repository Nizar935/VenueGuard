package com.nizar.venueguard;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

import static org.bytedeco.opencv.global.opencv_highgui.WINDOW_AUTOSIZE;
import static org.bytedeco.opencv.global.opencv_highgui.destroyAllWindows;
import static org.bytedeco.opencv.global.opencv_highgui.imshow;
import static org.bytedeco.opencv.global.opencv_highgui.namedWindow;
import static org.bytedeco.opencv.global.opencv_highgui.waitKey;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;

public final class CameraPreviewRunner {

    private static final int CAMERA_INDEX = 0;

    private static final int REQUESTED_WIDTH = 1280;
    private static final int REQUESTED_HEIGHT = 720;

    private static final String WINDOW_NAME =
            "VenueGuard Camera Preview";

    private CameraPreviewRunner() {
    }

    public static void main(String[] args) {
        System.out.println("VenueGuard camera preview starting...");
        System.out.println("Opening camera index: " + CAMERA_INDEX);

        try (
                VideoCapture camera =
                        new VideoCapture(CAMERA_INDEX);

                Mat frame = new Mat()
        ) {
            if (!camera.isOpened()) {
                throw new IllegalStateException(
                        "Could not open camera index "
                                + CAMERA_INDEX
                );
            }

            camera.set(
                    CAP_PROP_FRAME_WIDTH,
                    REQUESTED_WIDTH
            );

            camera.set(
                    CAP_PROP_FRAME_HEIGHT,
                    REQUESTED_HEIGHT
            );

            System.out.printf(
                    "Camera opened: %.0f × %.0f%n",
                    camera.get(CAP_PROP_FRAME_WIDTH),
                    camera.get(CAP_PROP_FRAME_HEIGHT)
            );

            System.out.println(
                    "Press Q or Escape in the camera window to stop."
            );

            namedWindow(
                    WINDOW_NAME,
                    WINDOW_AUTOSIZE
            );

            while (true) {
                boolean frameCaptured = camera.read(frame);

                if (!frameCaptured || frame.empty()) {
                    System.err.println(
                            "The camera did not return a valid frame."
                    );
                    break;
                }

                imshow(WINDOW_NAME, frame);

                int pressedKey = waitKey(1) & 0xFF;

                if (
                        pressedKey == 'q'
                                || pressedKey == 'Q'
                                || pressedKey == 27
                ) {
                    break;
                }
            }

            camera.release();

            System.out.println(
                    "VenueGuard camera preview stopped."
            );
        } finally {
            destroyAllWindows();
        }
    }
}