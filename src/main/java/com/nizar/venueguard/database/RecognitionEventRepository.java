package com.nizar.venueguard.database;

import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Types;
import java.util.Objects;

public final class RecognitionEventRepository {

    private static final String INSERT_EVENT_SQL =
            """
            INSERT INTO recognition_events (
                occurred_at_utc,
                identity_label,
                recognized,
                similarity,
                threshold,
                camera_index,
                decision_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """;

    private final VenueGuardDatabase database;

    public RecognitionEventRepository(
            VenueGuardDatabase database
    ) {
        this.database =
                Objects.requireNonNull(
                        database,
                        "Database cannot be null"
                );
    }

    public void save(
            RecognitionEvent event
    ) throws IOException, SQLException {
        Objects.requireNonNull(
                event,
                "Recognition event cannot be null"
        );

        try (
                Connection connection =
                        database.openConnection();

                PreparedStatement statement =
                        connection.prepareStatement(
                                INSERT_EVENT_SQL
                        )
        ) {
            statement.setString(
                    1,
                    event.occurredAtUtc().toString()
            );

            statement.setString(
                    2,
                    event.identityLabel()
            );

            statement.setInt(
                    3,
                    event.recognized() ? 1 : 0
            );

            if (Double.isFinite(event.similarity())) {
                statement.setDouble(
                        4,
                        event.similarity()
                );
            } else {
                statement.setNull(
                        4,
                        Types.REAL
                );
            }

            statement.setDouble(
                    5,
                    event.threshold()
            );

            statement.setInt(
                    6,
                    event.cameraIndex()
            );

            statement.setString(
                    7,
                    event.decisionReason()
            );

            int affectedRows =
                    statement.executeUpdate();

            if (affectedRows != 1) {
                throw new SQLException(
                        "Expected to insert one recognition event, "
                                + "but inserted "
                                + affectedRows
                );
            }
        }
    }

    public long count()
            throws IOException, SQLException {
        try (
                Connection connection =
                        database.openConnection();

                PreparedStatement statement =
                        connection.prepareStatement(
                                """
                                SELECT COUNT(*)
                                FROM recognition_events
                                """
                        );

                ResultSet results =
                        statement.executeQuery()
        ) {
            return results.next()
                    ? results.getLong(1)
                    : 0;
        }
    }
}