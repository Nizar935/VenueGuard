package com.nizar.venueguard.enrollment;

import java.nio.file.Path;
import java.util.Objects;

public record FaceTemplate(
        Path sourceImage,
        float[] embedding
) {

    private static final int EXPECTED_DIMENSIONS = 128;

    public FaceTemplate {
        Objects.requireNonNull(
                sourceImage,
                "Source image cannot be null"
        );

        Objects.requireNonNull(
                embedding,
                "Embedding cannot be null"
        );

        sourceImage =
                sourceImage.toAbsolutePath().normalize();

        embedding = embedding.clone();

        if (embedding.length != EXPECTED_DIMENSIONS) {
            throw new IllegalArgumentException(
                    "Expected a 128-value embedding but received "
                            + embedding.length
            );
        }

        for (float value : embedding) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(
                        "Embedding contains a non-finite value"
                );
            }
        }
    }

    @Override
    public float[] embedding() {
        return embedding.clone();
    }

    public int dimensions() {
        return embedding.length;
    }
}
