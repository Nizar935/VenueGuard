package com.nizar.venueguard.vision;

import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.Objects;

import static org.bytedeco.opencv.global.opencv_core.CV_32FC1;

public record FaceDetection(
        float x,
        float y,
        float width,
        float height,
        Landmark rightEye,
        Landmark leftEye,
        Landmark nose,
        Landmark rightMouth,
        Landmark leftMouth,
        float confidence
) {

    private static final int YUNET_VALUE_COUNT = 15;

    public FaceDetection {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException(
                    "Face dimensions must be positive"
            );
        }

        Objects.requireNonNull(
                rightEye,
                "Right-eye landmark cannot be null"
        );

        Objects.requireNonNull(
                leftEye,
                "Left-eye landmark cannot be null"
        );

        Objects.requireNonNull(
                nose,
                "Nose landmark cannot be null"
        );

        Objects.requireNonNull(
                rightMouth,
                "Right-mouth landmark cannot be null"
        );

        Objects.requireNonNull(
                leftMouth,
                "Left-mouth landmark cannot be null"
        );
    }

    public FaceDetection scaledBy(float scale) {
        if (!Float.isFinite(scale) || scale <= 0) {
            throw new IllegalArgumentException(
                    "Scale must be positive and finite"
            );
        }

        return new FaceDetection(
                x * scale,
                y * scale,
                width * scale,
                height * scale,
                rightEye.scaledBy(scale),
                leftEye.scaledBy(scale),
                nose.scaledBy(scale),
                rightMouth.scaledBy(scale),
                leftMouth.scaledBy(scale),
                confidence
        );
    }

    /**
     * Converts this detection into the 1 × 15 matrix expected
     * by OpenCV's FaceRecognizerSF.
     */
    public Mat toOpenCvRow() {
        Mat row = new Mat(
                1,
                YUNET_VALUE_COUNT,
                CV_32FC1
        );

        try (FloatIndexer indexer = row.createIndexer()) {
            indexer.put(0, 0, x);
            indexer.put(0, 1, y);
            indexer.put(0, 2, width);
            indexer.put(0, 3, height);

            indexer.put(0, 4, rightEye.x());
            indexer.put(0, 5, rightEye.y());

            indexer.put(0, 6, leftEye.x());
            indexer.put(0, 7, leftEye.y());

            indexer.put(0, 8, nose.x());
            indexer.put(0, 9, nose.y());

            indexer.put(0, 10, rightMouth.x());
            indexer.put(0, 11, rightMouth.y());

            indexer.put(0, 12, leftMouth.x());
            indexer.put(0, 13, leftMouth.y());

            indexer.put(0, 14, confidence);
        }

        return row;
    }

    public record Landmark(float x, float y) {

        public Landmark scaledBy(float scale) {
            return new Landmark(
                    x * scale,
                    y * scale
            );
        }
    }
}
