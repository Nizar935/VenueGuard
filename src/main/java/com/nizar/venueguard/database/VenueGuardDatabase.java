package com.nizar.venueguard.database;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Objects;

public final class VenueGuardDatabase {

    private static final Path DEFAULT_DATABASE_PATH =
            Path.of(
                    "data",
                    "database",
                    "venueguard.db"
            );

    private final Path databasePath;

    public VenueGuardDatabase() {
        this(DEFAULT_DATABASE_PATH);
    }

    public VenueGuardDatabase(Path databasePath) {
        Objects.requireNonNull(
                databasePath,
                "Database path cannot be null"
        );

        this.databasePath =
                databasePath
                        .toAbsolutePath()
                        .normalize();
    }

    public Connection openConnection()
            throws IOException, SQLException {
        Path parentDirectory =
                databasePath.getParent();

        if (parentDirectory != null) {
            Files.createDirectories(
                    parentDirectory
            );
        }

        String connectionUrl =
                "jdbc:sqlite:" + databasePath;

        Connection connection =
                DriverManager.getConnection(
                        connectionUrl
                );

        try {
            configureConnection(connection);
            return connection;
        } catch (SQLException exception) {
            connection.close();
            throw exception;
        }
    }

    public void initialize()
            throws IOException, SQLException {
        try (
                Connection connection =
                        openConnection();

                Statement statement =
                        connection.createStatement()
        ) {
            statement.executeUpdate(
                    """
                    CREATE TABLE IF NOT EXISTS identities (
                        identity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_name TEXT NOT NULL
                            COLLATE NOCASE
                            UNIQUE,
                        profile_path TEXT NOT NULL UNIQUE,
                        enabled INTEGER NOT NULL DEFAULT 1
                            CHECK (enabled IN (0, 1)),
                        registered_at_utc TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL
                    )
                    """
            );

            statement.executeUpdate(
                    """
                    CREATE INDEX IF NOT EXISTS
                        idx_identities_enabled
                    ON identities (
                        enabled
                    )
                    """
            );

            statement.executeUpdate(
                    """
                    CREATE TABLE IF NOT EXISTS recognition_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        occurred_at_utc TEXT NOT NULL,
                        identity_label TEXT NOT NULL,
                        recognized INTEGER NOT NULL
                            CHECK (recognized IN (0, 1)),
                        similarity REAL,
                        threshold REAL NOT NULL,
                        camera_index INTEGER NOT NULL,
                        decision_reason TEXT NOT NULL
                    )
                    """
            );

            statement.executeUpdate(
                    """
                    CREATE INDEX IF NOT EXISTS
                        idx_recognition_events_time
                    ON recognition_events (
                        occurred_at_utc
                    )
                    """
            );

            statement.executeUpdate(
                    """
                    CREATE INDEX IF NOT EXISTS
                        idx_recognition_events_identity
                    ON recognition_events (
                        identity_label
                    )
                    """
            );

            statement.executeUpdate(
                    """
                    CREATE INDEX IF NOT EXISTS
                        idx_recognition_events_recognized
                    ON recognition_events (
                        recognized
                    )
                    """
            );
        }
    }

    public Path databasePath() {
        return databasePath;
    }

    private static void configureConnection(
            Connection connection
    ) throws SQLException {
        try (
                Statement statement =
                        connection.createStatement()
        ) {
            statement.execute(
                    "PRAGMA foreign_keys = ON"
            );

            statement.execute(
                    "PRAGMA busy_timeout = 5000"
            );

            statement.execute(
                    "PRAGMA journal_mode = WAL"
            );
        }
    }
}