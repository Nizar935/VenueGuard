package com.nizar.venueguard.enrollment;

import java.util.List;
import java.util.Objects;

public record EnrollmentProfile(
        String personName,
        List<FaceTemplate> templates
) {

    public EnrollmentProfile {
        Objects.requireNonNull(
                personName,
                "Person name cannot be null"
        );

        Objects.requireNonNull(
                templates,
                "Templates cannot be null"
        );

        personName = personName.trim();
        templates = List.copyOf(templates);

        if (personName.isBlank()) {
            throw new IllegalArgumentException(
                    "Person name cannot be blank"
            );
        }

        if (templates.isEmpty()) {
            throw new IllegalArgumentException(
                    "An enrolment profile requires "
                            + "at least one face template"
            );
        }
    }

    public int sampleCount() {
        return templates.size();
    }
}
