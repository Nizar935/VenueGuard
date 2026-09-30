package com.nizar.venueguard.vision;

import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.FaceDetectorYN;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class YuNetFaceDetector implements AutoCloseable {

    private static final int INITIAL_INPUT_WIDTH = 320;
    private static final int INITIAL_INPUT_HEIGHT = 320;

    private final FaceDetectorYN detector;

    public YuNetFaceDetector(Path modelPath) {
        Objects.requireNonNull(
                modelPath,
                "Model path cannot be null"
        );

        Path absoluteModelPath =
                modelPath.toAbsolutePath().normalize();

        if (!Files.isRegularFile(absoluteModelPath)) {
            throw new IllegalArgumentException(
                    "YuNet model does not exist: "
                            + absoluteModelPath
            );
        }

        try (
                Size initialInputSize = new Size(
                        INITIAL_INPUT_WIDTH,
                        INITIAL_INPUT_HEIGHT
                )
        ) {
            detector = FaceDetectorYN.create(
                    absoluteModelPath.toString(),
                    "",
                    initialInputSize
            );
        }

        if (detector == null || detector.isNull()) {
            throw new IllegalStateException(
                    "OpenCV could not create the YuNet detector"
            );
        }
    }

    public List<FaceDetection> detect(Mat image) {
        validateImage(image);

        try (
                Size actualInputSize = new Size(
                        image.cols(),
                        image.rows()
                )
        ) {
            detector.setInputSize(actualInputSize);
        }

        try (Mat rawDetections = new Mat()) {
            detector.detect(image, rawDetections);

            if (rawDetections.empty()) {
                return List.of();
            }

            return convertDetections(rawDetections);
        }
    }

    private List<FaceDetection> convertDetections(
            Mat rawDetections
    ) {
        List<FaceDetection> detections =
                new ArrayList<>(rawDetections.rows());

        try (
                FloatIndexer values =
                        rawDetections.createIndexer()
        ) {
            for (long row = 0; row < rawDetections.rows(); row++) {
                detections.add(
                        new FaceDetection(
                                values.get(row, 0),
                                values.get(row, 1),
                                values.get(row, 2),
                                values.get(row, 3),

                                landmark(values, row, 4),
                                landmark(values, row, 6),
                                landmark(values, row, 8),
                                landmark(values, row, 10),
                                landmark(values, row, 12),

                                values.get(row, 14)
                        )
                );
            }
        }

        return List.copyOf(detections);
    }

    private FaceDetection.Landmark landmark(
            FloatIndexer values,
            long row,
            long startingColumn
    ) {
        return new FaceDetection.Landmark(
                values.get(row, startingColumn),
                values.get(row, startingColumn + 1)
        );
    }

    private void validateImage(Mat image) {
        if (image == null || image.empty()) {
            throw new IllegalArgumentException(
                    "Detection image cannot be null or empty"
            );
        }

        if (image.channels() != 3) {
            throw new IllegalArgumentException(
                    "YuNet expects a three-channel colour image"
            );
        }
    }

    @Override
    public void close() {
        detector.close();
    }
}
