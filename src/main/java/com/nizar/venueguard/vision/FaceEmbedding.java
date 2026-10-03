package com.nizar.venueguard.vision;

import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.Objects;

public record FaceEmbedding(
        Mat alignedFace,
        Mat values
) implements AutoCloseable {

    public FaceEmbedding {
        Objects.requireNonNull(
                alignedFace,
                "Aligned face cannot be null"
        );

        Objects.requireNonNull(
                values,
                "Embedding values cannot be null"
        );

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

    /**
     * Copies the native OpenCV embedding into normal Java memory.
     */
    public float[] copyValues() {
        if (values.rows() != 1) {
            throw new IllegalStateException(
                    "Expected a single-row embedding but received "
                            + values.rows() + " rows"
            );
        }

        if (values.channels() != 1) {
            throw new IllegalStateException(
                    "Expected a single-channel embedding but received "
                            + values.channels() + " channels"
            );
        }

        int valueCount =
                Math.toIntExact(dimensions());

        float[] copiedValues =
                new float[valueCount];

        try (
                FloatIndexer indexer =
                        values.createIndexer()
        ) {
            indexer.get(
                    0,
                    0,
                    copiedValues,
                    0,
                    copiedValues.length
            );
        }

        return copiedValues;
    }

    @Override
    public void close() {
        values.close();
        alignedFace.close();
    }
}