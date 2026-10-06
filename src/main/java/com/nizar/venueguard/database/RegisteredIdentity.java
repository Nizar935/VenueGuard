package com.nizar.venueguard.database;

import java.nio.file.Path;
import java.time.Instant;
import java.util.Objects;

public record RegisteredIdentity(
        long identityId,
        String personName,
        Path profilePath,
        boolean enabled,
        Instant registeredAtUtc,
        Instant updatedAtUtc
) {

    public RegisteredIdentity {
        if (identityId <= 0) {
            throw new IllegalArgumentException(
                    "Identity ID must be positive"
            );
        }

        Objects.requireNonNull(
                personName,
                "Person name cannot be null"
        );

        Objects.requireNonNull(
                profilePath,
                "Profile path cannot be null"
        );

        Objects.requireNonNull(
                registeredAtUtc,
                "Registration time cannot be null"
        );

        Objects.requireNonNull(
                updatedAtUtc,
                "Update time cannot be null"
        );

        personName =
                personName.trim();

        profilePath =
                profilePath.normalize();

        if (personName.isBlank()) {
            throw new IllegalArgumentException(
                    "Person name cannot be blank"
            );
        }

        if (profilePath.toString().isBlank()) {
            throw new IllegalArgumentException(
                    "Profile path cannot be blank"
            );
        }

        if (
                updatedAtUtc.isBefore(
                        registeredAtUtc
                )
        ) {
            throw new IllegalArgumentException(
                    "Update time cannot precede registration time"
            );
        }
    }
}