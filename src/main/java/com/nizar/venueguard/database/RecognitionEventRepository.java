package com.nizar.venueguard.database;

import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Types;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
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

    private static final String FIND_RECENT_EVENTS_SQL =
            """
            SELECT
                event_id,
                occurred_at_utc,
                identity_label,
                recognized,
                similarity,
                threshold,
                camera_index,
                decision_reason
            FROM recognition_events
            ORDER BY
                occurred_at_utc DESC,
                event_id DESC
            LIMIT ?
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

    public List<StoredRecognitionEvent> findRecent(
            int maximumResults
    ) throws IOException, SQLException {
        if (maximumResults <= 0) {
            throw new IllegalArgumentException(
                    "Maximum results must be positive"
            );
        }

        List<StoredRecognitionEvent> events =
                new ArrayList<>();

        try (
                Connection connection =
                        database.openConnection();

                PreparedStatement statement =
                        connection.prepareStatement(
                                FIND_RECENT_EVENTS_SQL
                        )
        ) {
            statement.setInt(
                    1,
                    maximumResults
            );

            try (
                    ResultSet results =
                            statement.executeQuery()
            ) {
                while (results.next()) {
                    events.add(
                            readEvent(results)
                    );
                }
            }
        }

        return List.copyOf(events);
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

    private static StoredRecognitionEvent readEvent(
            ResultSet results
    ) throws SQLException {
        double similarityValue =
                results.getDouble("similarity");

        Double similarity =
                results.wasNull()
                        ? null
                        : similarityValue;

        return new StoredRecognitionEvent(
                results.getLong("event_id"),
                Instant.parse(
                        results.getString(
                                "occurred_at_utc"
                        )
                ),
                results.getString(
                        "identity_label"
                ),
                results.getInt(
                        "recognized"
                ) == 1,
                similarity,
                results.getDouble(
                        "threshold"
                ),
                results.getInt(
                        "camera_index"
                ),
                results.getString(
                        "decision_reason"
                )
        );
    }
}