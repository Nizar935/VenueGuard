package com.nizar.venueguard.database;

import java.time.Instant;
import java.util.Objects;

public record StoredRecognitionEvent(
        long eventId,
        Instant occurredAtUtc,
        String identityLabel,
        boolean recognized,
        Double similarity,
        double threshold,
        int cameraIndex,
        String decisionReason
) {

    public StoredRecognitionEvent {
        if (eventId <= 0) {
            throw new IllegalArgumentException(
                    "Event ID must be positive"
            );
        }

        Objects.requireNonNull(
                occurredAtUtc,
                "Event time cannot be null"
        );

        Objects.requireNonNull(
                identityLabel,
                "Identity label cannot be null"
        );

        Objects.requireNonNull(
                decisionReason,
                "Decision reason cannot be null"
        );

        identityLabel =
                identityLabel.trim();

        decisionReason =
                decisionReason.trim();

        if (identityLabel.isBlank()) {
            throw new IllegalArgumentException(
                    "Identity label cannot be blank"
            );
        }

        if (
                similarity != null
                        && !Double.isFinite(similarity)
        ) {
            throw new IllegalArgumentException(
                    "Similarity must be finite when present"
            );
        }

        if (
                !Double.isFinite(threshold)
                        || threshold < -1
                        || threshold > 1
        ) {
            throw new IllegalArgumentException(
                    "Threshold must be between -1 and 1"
            );
        }

        if (cameraIndex < 0) {
            throw new IllegalArgumentException(
                    "Camera index cannot be negative"
            );
        }

        if (decisionReason.isBlank()) {
            throw new IllegalArgumentException(
                    "Decision reason cannot be blank"
            );
        }
    }
}
