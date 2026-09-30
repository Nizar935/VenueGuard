package com.nizar.venueguard.vision;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

public final class SFaceEmbedder implements AutoCloseable {

    private final FaceRecognizerSF recognizer;

    public SFaceEmbedder(Path modelPath) {
        Objects.requireNonNull(
                modelPath,
                "SFace model path cannot be null"
        );

        Path absoluteModelPath =
                modelPath.toAbsolutePath().normalize();

        if (!Files.isRegularFile(absoluteModelPath)) {
            throw new IllegalArgumentException(
                    "SFace model does not exist: "
                            + absoluteModelPath
            );
        }

        recognizer = FaceRecognizerSF.create(
                absoluteModelPath.toString(),
                ""
        );

        if (recognizer == null || recognizer.isNull()) {
            throw new IllegalStateException(
                    "OpenCV could not load the SFace model: "
                            + absoluteModelPath
            );
        }
    }

    public FaceEmbedding createEmbedding(
            Mat sourceImage,
            Mat detectedFace
    ) {
        Objects.requireNonNull(
                sourceImage,
                "Source image cannot be null"
        );

        Objects.requireNonNull(
                detectedFace,
                "Detected face cannot be null"
        );

        if (sourceImage.empty()) {
            throw new IllegalArgumentException(
                    "Source image cannot be empty"
            );
        }

        if (detectedFace.empty()) {
            throw new IllegalArgumentException(
                    "Detected face data cannot be empty"
            );
        }

        Mat alignedFace = new Mat();
        Mat embeddingValues = new Mat();

        try {
            recognizer.alignCrop(
                    sourceImage,
                    detectedFace,
                    alignedFace
            );

            if (alignedFace.empty()) {
                throw new IllegalStateException(
                        "SFace failed to align the detected face"
                );
            }

            recognizer.feature(
                    alignedFace,
                    embeddingValues
            );

            if (embeddingValues.empty()) {
                throw new IllegalStateException(
                        "SFace failed to generate an embedding"
                );
            }

            return new FaceEmbedding(
                    alignedFace,
                    embeddingValues
            );
        } catch (RuntimeException exception) {
            embeddingValues.close();
            alignedFace.close();
            throw exception;
        }
    }

    @Override
    public void close() {
        recognizer.close();
    }
}
