package com.nizar.venueguard;

import com.nizar.venueguard.enrollment.EnrollmentProfile;
import com.nizar.venueguard.enrollment.EnrollmentReport;
import com.nizar.venueguard.enrollment.FaceEnrollmentService;
import com.nizar.venueguard.enrollment.FaceTemplate;
import com.nizar.venueguard.enrollment.RejectedEnrollmentImage;
import com.nizar.venueguard.vision.SFaceEmbedder;
import com.nizar.venueguard.vision.YuNetFaceDetector;
import com.nizar.venueguard.enrollment.EnrollmentProfileStore;

import java.io.IOException;
import java.nio.file.Path;

public final class EnrollmentRunner {

    private static final Path ENROLLMENT_DIRECTORY =
            Path.of(
                    "data",
                    "enrollment",
                    "Nizar"
            );
    private static final Path PROFILE_OUTPUT =
            Path.of(
                    "data",
                    "profiles",
                    "Nizar.vgp"
            );

    private static final Path FACE_DETECTION_MODEL =
            Path.of(
                    "models",
                    "face_detection_yunet_2023mar.onnx"
            );

    private static final Path FACE_RECOGNITION_MODEL =
            Path.of(
                    "models",
                    "face_recognition_sface_2021dec.onnx"
            );

    private static final int MAXIMUM_IMAGE_DIMENSION = 480;

    private EnrollmentRunner() {
    }

    public static void main(String[] args) throws IOException {
        System.out.println("VenueGuard enrolment starting...");
        System.out.println(
                "Enrollment directory: "
                        + ENROLLMENT_DIRECTORY
                        .toAbsolutePath()
                        .normalize()
        );

        try (
                YuNetFaceDetector detector =
                        new YuNetFaceDetector(
                                FACE_DETECTION_MODEL
                        );

                SFaceEmbedder embedder =
                        new SFaceEmbedder(
                                FACE_RECOGNITION_MODEL
                        )
        ) {
            FaceEnrollmentService enrollmentService =
                    new FaceEnrollmentService(
                            MAXIMUM_IMAGE_DIMENSION,
                            detector,
                            embedder
                    );

            EnrollmentReport report =
                    enrollmentService.enroll(
                            ENROLLMENT_DIRECTORY
                    );

            printReport(report);

            EnrollmentProfileStore profileStore =
                    new EnrollmentProfileStore();

            profileStore.save(
                    report.profile(),
                    PROFILE_OUTPUT
            );

            System.out.println();
            System.out.println(
                    "Profile saved to: "
                            + PROFILE_OUTPUT
                            .toAbsolutePath()
                            .normalize()
            );

        }

        System.out.println();
        System.out.println(
                "Enrollment processing completed successfully!"
        );
    }

    private static void printReport(
            EnrollmentReport report
    ) {
        EnrollmentProfile profile =
                report.profile();

        System.out.println();
        System.out.println("Enrollment report");
        System.out.println("-----------------");

        System.out.println(
                "Person: " + profile.personName()
        );

        System.out.println(
                "Photographs processed: "
                        + report.totalCount()
        );

        System.out.println(
                "Photographs accepted: "
                        + report.acceptedCount()
        );

        System.out.println(
                "Photographs rejected: "
                        + report.rejectedCount()
        );

        System.out.println();
        System.out.println("Accepted photographs:");

        for (FaceTemplate template : profile.templates()) {
            System.out.printf(
                    "  ACCEPTED: %s (%d embedding values)%n",
                    template.sourceImage()
                            .getFileName(),
                    template.dimensions()
            );
        }

        if (!report.rejectedImages().isEmpty()) {
            System.out.println();
            System.out.println("Rejected photographs:");

            for (
                    RejectedEnrollmentImage rejected
                    : report.rejectedImages()
            ) {
                System.out.printf(
                        "  REJECTED: %s%n",
                        rejected.sourceImage()
                                .getFileName()
                );

                System.out.println(
                        "    Reason: " + rejected.reason()
                );
            }
        }

        System.out.println();
        System.out.println(
                "Profile templates created: "
                        + profile.sampleCount()
        );

        System.out.println(
                "Status: ENROLMENT PROFILE READY"
        );
    }
}