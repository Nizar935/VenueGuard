package com.nizar.venueguard.enrollment;

import java.util.List;
import java.util.Objects;

public record EnrollmentReport(
        EnrollmentProfile profile,
        List<RejectedEnrollmentImage> rejectedImages
) {

    public EnrollmentReport {
        Objects.requireNonNull(
                profile,
                "Enrollment profile cannot be null"
        );

        Objects.requireNonNull(
                rejectedImages,
                "Rejected images cannot be null"
        );

        rejectedImages =
                List.copyOf(rejectedImages);
    }

    public int acceptedCount() {
        return profile.sampleCount();
    }

    public int rejectedCount() {
        return rejectedImages.size();
    }

    public int totalCount() {
        return acceptedCount() + rejectedCount();
    }
}