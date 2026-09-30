package com.nizar.venueguard.vision;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;

import java.util.List;
import java.util.Objects;

import static org.bytedeco.opencv.global.opencv_imgproc.FILLED;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_AA;
import static org.bytedeco.opencv.global.opencv_imgproc.circle;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;

public final class FaceRenderer {

    private FaceRenderer() {
    }

    public static Mat render(
            Mat source,
            List<FaceDetection> detections
    ) {
        Objects.requireNonNull(source);
        Objects.requireNonNull(detections);

        if (source.empty()) {
            throw new IllegalArgumentException(
                    "Source image cannot be empty"
            );
        }

        Mat output = new Mat();
        source.copyTo(output);

        try (
                Scalar boxColour = new Scalar(0, 255, 0, 0);
                Scalar landmarkColour = new Scalar(0, 0, 255, 0)
        ) {
            for (FaceDetection detection : detections) {
                drawBox(output, detection, boxColour);
                drawLandmarks(
                        output,
                        detection,
                        landmarkColour
                );
            }

            return output;
        } catch (RuntimeException exception) {
            output.close();
            throw exception;
        }
    }

    private static void drawBox(
            Mat image,
            FaceDetection detection,
            Scalar colour
    ) {
        int x = clamp(
                Math.round(detection.x()),
                0,
                image.cols() - 1
        );

        int y = clamp(
                Math.round(detection.y()),
                0,
                image.rows() - 1
        );

        int width = Math.min(
                Math.max(1, Math.round(detection.width())),
                image.cols() - x
        );

        int height = Math.min(
                Math.max(1, Math.round(detection.height())),
                image.rows() - y
        );

        try (Rect box = new Rect(x, y, width, height)) {
            rectangle(
                    image,
                    box,
                    colour,
                    2,
                    LINE_AA,
                    0
            );
        }
    }

    private static void drawLandmarks(
            Mat image,
            FaceDetection detection,
            Scalar colour
    ) {
        drawLandmark(image, detection.rightEye(), colour);
        drawLandmark(image, detection.leftEye(), colour);
        drawLandmark(image, detection.nose(), colour);
        drawLandmark(image, detection.rightMouth(), colour);
        drawLandmark(image, detection.leftMouth(), colour);
    }

    private static void drawLandmark(
            Mat image,
            FaceDetection.Landmark landmark,
            Scalar colour
    ) {
        int x = clamp(
                Math.round(landmark.x()),
                0,
                image.cols() - 1
        );

        int y = clamp(
                Math.round(landmark.y()),
                0,
                image.rows() - 1
        );

        try (Point centre = new Point(x, y)) {
            circle(
                    image,
                    centre,
                    3,
                    colour,
                    FILLED,
                    LINE_AA,
                    0
            );
        }
    }

    private static int clamp(
            int value,
            int minimum,
            int maximum
    ) {
        return Math.max(
                minimum,
                Math.min(value, maximum)
        );
    }
}
