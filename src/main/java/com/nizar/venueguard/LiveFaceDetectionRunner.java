package com.nizar.venueguard;

import com.nizar.venueguard.vision.FaceDetection;
import com.nizar.venueguard.vision.FaceRenderer;
import com.nizar.venueguard.vision.ImagePreprocessor;
import com.nizar.venueguard.vision.PreparedImage;
import com.nizar.venueguard.vision.YuNetFaceDetector;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

import java.nio.file.Path;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_highgui.WINDOW_AUTOSIZE;
import static org.bytedeco.opencv.global.opencv_highgui.destroyAllWindows;
import static org.bytedeco.opencv.global.opencv_highgui.imshow;
import static org.bytedeco.opencv.global.opencv_highgui.namedWindow;
import static org.bytedeco.opencv.global.opencv_highgui.waitKey;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;

public final class LiveFaceDetectionRunner {

    private static final int CAMERA_INDEX = 0;

    private static final int REQUESTED_WIDTH = 1280;
    private static final int REQUESTED_HEIGHT = 720;

    private static final int MAXIMUM_IMAGE_DIMENSION = 480;

    private static final Path FACE_DETECTION_MODEL =
            Path.of(
                    "models",
                    "face_detection_yunet_2023mar.onnx"
            );

    private static final String WINDOW_NAME =
            "VenueGuard Live Face Detection";

    private LiveFaceDetectionRunner() {
    }

    public static void main(String[] args) {
        System.out.println(
                "VenueGuard live face detection starting..."
        );

        ImagePreprocessor preprocessor =
                new ImagePreprocessor(
                        MAXIMUM_IMAGE_DIMENSION
                );

        try (
                VideoCapture camera =
                        new VideoCapture(CAMERA_INDEX);

                YuNetFaceDetector detector =
                        new YuNetFaceDetector(
                                FACE_DETECTION_MODEL
                        );

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
                boolean frameCaptured =
                        camera.read(frame);

                if (!frameCaptured || frame.empty()) {
                    System.err.println(
                            "The camera did not return a valid frame."
                    );

                    break;
                }

                try (
                        PreparedImage prepared =
                                preprocessor.preprocess(frame)
                ) {
                    List<FaceDetection> preparedDetections =
                            detector.detect(
                                    prepared.colour()
                            );

                    float originalScale =
                            (float) (
                                    1.0
                                            / prepared.scaleFactor()
                            );

                    List<FaceDetection> frameDetections =
                            preparedDetections
                                    .stream()
                                    .map(
                                            detection ->
                                                    detection.scaledBy(
                                                            originalScale
                                                    )
                                    )
                                    .toList();

                    try (
                            Mat renderedFrame =
                                    FaceRenderer.render(
                                            frame,
                                            frameDetections
                                    )
                    ) {
                        imshow(
                                WINDOW_NAME,
                                renderedFrame
                        );
                    }
                }

                int pressedKey =
                        waitKey(1) & 0xFF;

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
                    "VenueGuard live face detection stopped."
            );
        } finally {
            destroyAllWindows();
        }
    }
}
