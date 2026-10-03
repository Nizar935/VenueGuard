package com.nizar.venueguard.enrollment;

import java.nio.file.Path;
import java.util.Objects;

public record RejectedEnrollmentImage(
        Path sourceImage,
        String reason
) {

    public RejectedEnrollmentImage {
        Objects.requireNonNull(
                sourceImage,
                "Source image cannot be null"
        );

        Objects.requireNonNull(
                reason,
                "Rejection reason cannot be null"
        );

        sourceImage =
                sourceImage.toAbsolutePath().normalize();

        reason = reason.trim();

        if (reason.isBlank()) {
            throw new IllegalArgumentException(
                    "Rejection reason cannot be blank"
            );
        }
    }
}
