package com.nizar.venueguard.recognition;

import com.nizar.venueguard.database.IdentityRepository;
import com.nizar.venueguard.database.RegisteredIdentity;
import com.nizar.venueguard.enrollment.EnrollmentProfile;
import com.nizar.venueguard.enrollment.EnrollmentProfileStore;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class EnabledProfileLoader {

    private final IdentityRepository identityRepository;
    private final EnrollmentProfileStore profileStore;

    public EnabledProfileLoader(
            IdentityRepository identityRepository,
            EnrollmentProfileStore profileStore
    ) {
        this.identityRepository =
                Objects.requireNonNull(
                        identityRepository,
                        "Identity repository cannot be null"
                );

        this.profileStore =
                Objects.requireNonNull(
                        profileStore,
                        "Profile store cannot be null"
                );
    }

    public List<EnrollmentProfile> loadEnabledProfiles()
            throws IOException, SQLException {
        List<RegisteredIdentity> identities =
                identityRepository.findEnabled();

        List<EnrollmentProfile> profiles =
                new ArrayList<>();

        for (RegisteredIdentity identity : identities) {
            Path profilePath =
                    identity.profilePath();

            if (!Files.isRegularFile(profilePath)) {
                throw new IOException(
                        "Profile file does not exist for "
                                + identity.personName()
                                + ": "
                                + profilePath
                                .toAbsolutePath()
                                .normalize()
                );
            }

            EnrollmentProfile profile =
                    profileStore.load(profilePath);

            if (
                    !profile.personName()
                            .equalsIgnoreCase(
                                    identity.personName()
                            )
            ) {
                throw new IllegalStateException(
                        "Database identity name "
                                + identity.personName()
                                + " does not match profile name "
                                + profile.personName()
                );
            }

            profiles.add(profile);
        }

        return List.copyOf(profiles);
    }
}