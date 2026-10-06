package com.nizar.venueguard;

import com.nizar.venueguard.database.IdentityRepository;
import com.nizar.venueguard.database.RegisteredIdentity;
import com.nizar.venueguard.database.VenueGuardDatabase;

import java.io.IOException;
import java.nio.file.Path;
import java.sql.SQLException;
import java.util.List;

public final class IdentitySetupRunner {

    private static final String PERSON_NAME =
            "Nizar";

    private static final Path PROFILE_PATH =
            Path.of(
                    "data",
                    "profiles",
                    "Nizar.vgp"
            );

    private IdentitySetupRunner() {
    }

    public static void main(String[] args)
            throws IOException, SQLException {
        System.out.println(
                "VenueGuard identity setup starting..."
        );

        VenueGuardDatabase database =
                new VenueGuardDatabase();

        database.initialize();

        IdentityRepository repository =
                new IdentityRepository(
                        database
                );

        RegisteredIdentity registeredIdentity =
                repository.registerOrUpdate(
                        PERSON_NAME,
                        PROFILE_PATH
                );

        System.out.printf(
                "Registered identity: %s (ID %d)%n",
                registeredIdentity.personName(),
                registeredIdentity.identityId()
        );

        List<RegisteredIdentity> identities =
                repository.findAll();

        System.out.println();
        System.out.printf(
                "%-5s %-20s %-10s %s%n",
                "ID",
                "Name",
                "Enabled",
                "Profile"
        );

        System.out.println(
                "----------------------------------------------------------------"
        );

        for (RegisteredIdentity identity : identities) {
            System.out.printf(
                    "%-5d %-20s %-10s %s%n",
                    identity.identityId(),
                    identity.personName(),
                    identity.enabled() ? "YES" : "NO",
                    identity.profilePath()
            );
        }

        System.out.println();
        System.out.println(
                "Enabled identities: "
                        + repository.findEnabled().size()
        );
    }
}