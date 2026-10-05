package com.nizar.venueguard;

import com.nizar.venueguard.database.RecognitionEventRepository;
import com.nizar.venueguard.database.StoredRecognitionEvent;
import com.nizar.venueguard.database.VenueGuardDatabase;

import java.io.IOException;
import java.sql.SQLException;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Locale;

public final class RecognitionHistoryRunner {

    private static final int MAXIMUM_RESULTS = 20;

    private static final DateTimeFormatter TIME_FORMATTER =
            DateTimeFormatter
                    .ofPattern(
                            "dd MMM uuuu HH:mm:ss",
                            Locale.UK
                    )
                    .withZone(
                            ZoneId.systemDefault()
                    );

    private RecognitionHistoryRunner() {
    }

    public static void main(String[] args)
            throws IOException, SQLException {
        VenueGuardDatabase database =
                new VenueGuardDatabase();

        database.initialize();

        RecognitionEventRepository repository =
                new RecognitionEventRepository(
                        database
                );

        List<StoredRecognitionEvent> events =
                repository.findRecent(
                        MAXIMUM_RESULTS
                );

        System.out.println(
                "VenueGuard recognition history"
        );

        System.out.println(
                "Database: "
                        + database.databasePath()
        );

        System.out.println(
                "Total stored events: "
                        + repository.count()
        );

        if (events.isEmpty()) {
            System.out.println(
                    "No recognition events have been stored."
            );

            return;
        }

        System.out.println();
        System.out.printf(
                "%-5s %-21s %-12s %-12s %-11s %-10s %-7s%n",
                "ID",
                "Local time",
                "Identity",
                "Decision",
                "Similarity",
                "Threshold",
                "Camera"
        );

        System.out.println(
                "-------------------------------------------------------------------------------"
        );

        for (StoredRecognitionEvent event : events) {
            String decision =
                    event.recognized()
                            ? "RECOGNIZED"
                            : "UNKNOWN";

            String similarity =
                    event.similarity() == null
                            ? "N/A"
                            : String.format(
                            Locale.ROOT,
                            "%.4f",
                            event.similarity()
                    );

            System.out.printf(
                    "%-5d %-21s %-12s %-12s %-11s %-10.4f %-7d%n",
                    event.eventId(),
                    TIME_FORMATTER.format(
                            event.occurredAtUtc()
                    ),
                    event.identityLabel(),
                    decision,
                    similarity,
                    event.threshold(),
                    event.cameraIndex()
            );
        }
    }
}