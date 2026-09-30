package com.nizar.venueguard.vision;

import org.bytedeco.opencv.opencv_core.Mat;

import java.util.Objects;

public record FaceEmbedding(
        Mat alignedFace,
        Mat values
) implements AutoCloseable {

    public FaceEmbedding {
        Objects.requireNonNull(alignedFace, "Aligned face cannot be null");
        Objects.requireNonNull(values, "Embedding values cannot be null");

        if (alignedFace.empty()) {
            throw new IllegalArgumentException(
                    "Aligned face cannot be empty"
            );
        }

        if (values.empty()) {
            throw new IllegalArgumentException(
                    "Embedding values cannot be empty"
            );
        }
    }

    public long dimensions() {
        return values.total() * values.channels();
    }

    @Override
    public void close() {
        values.close();
        alignedFace.close();
    }
}