package com.nizar.venueguard.recognition;

import java.nio.file.Path;
import java.util.Objects;

public record RecognitionResult(
        String identity,
        boolean recognized,
        double similarity,
        double threshold,
        Path bestReference,
        int templatesCompared
) {

    public RecognitionResult {
        Objects.requireNonNull(
                identity,
                "Identity cannot be null"
        );

        Objects.requireNonNull(
                bestReference,
                "Best reference cannot be null"
        );

        identity = identity.trim();

        bestReference =
                bestReference.toAbsolutePath().normalize();

        if (identity.isBlank()) {
            throw new IllegalArgumentException(
                    "Identity cannot be blank"
            );
        }

        if (!Double.isFinite(similarity)) {
            throw new IllegalArgumentException(
                    "Similarity must be finite"
            );
        }

        if (!Double.isFinite(threshold)) {
            throw new IllegalArgumentException(
                    "Threshold must be finite"
            );
        }

        if (templatesCompared <= 0) {
            throw new IllegalArgumentException(
                    "At least one template must be compared"
            );
        }
    }
}
