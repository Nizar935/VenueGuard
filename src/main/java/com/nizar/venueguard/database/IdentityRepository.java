package com.nizar.venueguard.database;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

public final class IdentityRepository {

    private static final String REGISTER_IDENTITY_SQL =
            """
            INSERT INTO identities (
                person_name,
                profile_path,
                enabled,
                registered_at_utc,
                updated_at_utc
            )
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(person_name)
            DO UPDATE SET
                profile_path = excluded.profile_path,
                enabled = 1,
                updated_at_utc = excluded.updated_at_utc
            """;

    private static final String SELECT_COLUMNS =
            """
            SELECT
                identity_id,
                person_name,
                profile_path,
                enabled,
                registered_at_utc,
                updated_at_utc
            FROM identities
            """;

    private final VenueGuardDatabase database;

    public IdentityRepository(
            VenueGuardDatabase database
    ) {
        this.database =
                Objects.requireNonNull(
                        database,
                        "Database cannot be null"
                );
    }

    public RegisteredIdentity registerOrUpdate(
            String personName,
            Path profilePath
    ) throws IOException, SQLException {
        Objects.requireNonNull(
                personName,
                "Person name cannot be null"
        );

        Objects.requireNonNull(
                profilePath,
                "Profile path cannot be null"
        );

        String normalizedName =
                personName.trim();

        Path normalizedPath =
                profilePath.normalize();

        if (normalizedName.isBlank()) {
            throw new IllegalArgumentException(
                    "Person name cannot be blank"
            );
        }

        if (!Files.isRegularFile(normalizedPath)) {
            throw new IllegalArgumentException(
                    "Profile file does not exist: "
                            + normalizedPath
                            .toAbsolutePath()
                            .normalize()
            );
        }

        Instant currentTime =
                Instant.now();

        try (
                Connection connection =
                        database.openConnection();

                PreparedStatement statement =
                        connection.prepareStatement(
                                REGISTER_IDENTITY_SQL
                        )
        ) {
            statement.setString(
                    1,
                    normalizedName
            );

            statement.setString(
                    2,
                    normalizedPath.toString()
            );

            statement.setString(
                    3,
                    currentTime.toString()
            );

            statement.setString(
                    4,
                    currentTime.toString()
            );

            statement.executeUpdate();
        }

        return findByName(normalizedName)
                .orElseThrow(
                        () -> new SQLException(
                                "Registered identity could not be reloaded"
                        )
                );
    }

    public Optional<RegisteredIdentity> findByName(
            String personName
    ) throws IOException, SQLException {
        Objects.requireNonNull(
                personName,
                "Person name cannot be null"
        );

        String normalizedName =
                personName.trim();

        if (normalizedName.isBlank()) {
            throw new IllegalArgumentException(
                    "Person name cannot be blank"
            );
        }

        String sql =
                SELECT_COLUMNS
                        + """
                        
                        WHERE person_name = ?
                            COLLATE NOCASE
                        LIMIT 1
                        """;

        try (
                Connection connection =
                        database.openConnection();

                PreparedStatement statement =
                        connection.prepareStatement(sql)
        ) {
            statement.setString(
                    1,
                    normalizedName
            );

            try (
                    ResultSet results =
                            statement.executeQuery()
            ) {
                if (!results.next()) {
                    return Optional.empty();
                }

                return Optional.of(
                        readIdentity(results)
                );
            }
        }
    }

    public List<RegisteredIdentity> findAll()
            throws IOException, SQLException {
        String sql =
                SELECT_COLUMNS
                        + """
                        
                        ORDER BY person_name
                            COLLATE NOCASE
                        """;

        return executeIdentityQuery(sql);
    }

    public List<RegisteredIdentity> findEnabled()
            throws IOException, SQLException {
        String sql =
                SELECT_COLUMNS
                        + """
                        
                        WHERE enabled = 1
                        ORDER BY person_name
                            COLLATE NOCASE
                        """;

        return executeIdentityQuery(sql);
    }

    public void setEnabled(
            long identityId,
            boolean enabled
    ) throws IOException, SQLException {
        if (identityId <= 0) {
            throw new IllegalArgumentException(
                    "Identity ID must be positive"
            );
        }

        try (
                Connection connection =
                        database.openConnection();

                PreparedStatement statement =
                        connection.prepareStatement(
                                """
                                UPDATE identities
                                SET
                                    enabled = ?,
                                    updated_at_utc = ?
                                WHERE identity_id = ?
                                """
                        )
        ) {
            statement.setInt(
                    1,
                    enabled ? 1 : 0
            );

            statement.setString(
                    2,
                    Instant.now().toString()
            );

            statement.setLong(
                    3,
                    identityId
            );

            int affectedRows =
                    statement.executeUpdate();

            if (affectedRows != 1) {
                throw new SQLException(
                        "No identity exists with ID "
                                + identityId
                );
            }
        }
    }

    private List<RegisteredIdentity> executeIdentityQuery(
            String sql
    ) throws IOException, SQLException {
        List<RegisteredIdentity> identities =
                new ArrayList<>();

        try (
                Connection connection =
                        database.openConnection();

                PreparedStatement statement =
                        connection.prepareStatement(sql);

                ResultSet results =
                        statement.executeQuery()
        ) {
            while (results.next()) {
                identities.add(
                        readIdentity(results)
                );
            }
        }

        return List.copyOf(identities);
    }

    private static RegisteredIdentity readIdentity(
            ResultSet results
    ) throws SQLException {
        return new RegisteredIdentity(
                results.getLong("identity_id"),
                results.getString("person_name"),
                Path.of(
                        results.getString(
                                "profile_path"
                        )
                ),
                results.getInt("enabled") == 1,
                Instant.parse(
                        results.getString(
                                "registered_at_utc"
                        )
                ),
                Instant.parse(
                        results.getString(
                                "updated_at_utc"
                        )
                )
        );
    }
}
