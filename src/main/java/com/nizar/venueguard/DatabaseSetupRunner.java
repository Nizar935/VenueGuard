package com.nizar.venueguard;

import com.nizar.venueguard.database.VenueGuardDatabase;

import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public final class DatabaseSetupRunner {

    private DatabaseSetupRunner() {
    }

    public static void main(String[] args)
            throws IOException, SQLException {
        System.out.println(
                "VenueGuard database setup starting..."
        );

        VenueGuardDatabase database =
                new VenueGuardDatabase();

        database.initialize();

        System.out.println(
                "Database location: "
                        + database.databasePath()
        );

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
            int eventCount =
                    results.next()
                            ? results.getInt(1)
                            : 0;

            System.out.println(
                    "Stored recognition events: "
                            + eventCount
            );
        }

        System.out.println(
                "VenueGuard database setup complete."
        );
    }
}
