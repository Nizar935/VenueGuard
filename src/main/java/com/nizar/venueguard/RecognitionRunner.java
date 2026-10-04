package com.nizar.venueguard;

import com.nizar.venueguard.enrollment.EnrollmentProfile;
import com.nizar.venueguard.enrollment.EnrollmentProfileStore;
import com.nizar.venueguard.recognition.FaceRecognitionService;
import com.nizar.venueguard.recognition.RecognitionResult;
import com.nizar.venueguard.vision.SFaceEmbedder;
import com.nizar.venueguard.vision.YuNetFaceDetector;

import java.io.IOException;
import java.nio.file.Path;

public final class RecognitionRunner {

    private static final Path QUERY_IMAGE =
            Path.of(
                    "data",
                    "input",
                    "sample.jpg"
                     //"unknown.png"
            );

    private static final Path PROFILE_PATH =
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

    private RecognitionRunner() {
    }

    public static void main(String[] args) throws IOException {
        System.out.println(
                "VenueGuard recognition starting..."
        );

        System.out.println(
                "Query image: "
                        + QUERY_IMAGE
                        .toAbsolutePath()
                        .normalize()
        );

        EnrollmentProfileStore profileStore =
                new EnrollmentProfileStore();

        EnrollmentProfile profile =
                profileStore.load(PROFILE_PATH);

        System.out.printf(
                "Loaded profile: %s (%d templates)%n",
                profile.personName(),
                profile.sampleCount()
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
            FaceRecognitionService recognitionService =
                    new FaceRecognitionService(
                            MAXIMUM_IMAGE_DIMENSION,
                            detector,
                            embedder
                    );

            RecognitionResult result =
                    recognitionService.recognize(
                            QUERY_IMAGE,
                            profile
                    );

            printResult(result);
        }
    }

    private static void printResult(
            RecognitionResult result
    ) {
        System.out.println();
        System.out.println("Recognition result");
        System.out.println("------------------");

        System.out.println(
                "Identity: " + result.identity()
        );

        System.out.println(
                "Decision: "
                        + (
                        result.recognized()
                                ? "RECOGNIZED"
                                : "UNKNOWN"
                )
        );

        System.out.printf(
                "Cosine similarity: %.4f%n",
                result.similarity()
        );

        System.out.printf(
                "Required threshold: %.4f%n",
                result.threshold()
        );

        System.out.println(
                "Templates compared: "
                        + result.templatesCompared()
        );

        System.out.println(
                "Best matching reference: "
                        + result.bestReference()
                        .getFileName()
        );
    }
}
