package com.nizar.venueguard;

import com.nizar.venueguard.database.RecognitionEvent;
import com.nizar.venueguard.database.RecognitionEventRepository;
import com.nizar.venueguard.database.VenueGuardDatabase;
import com.nizar.venueguard.enrollment.EnrollmentProfile;
import com.nizar.venueguard.enrollment.EnrollmentProfileStore;
import com.nizar.venueguard.recognition.FaceRecognitionService;
import com.nizar.venueguard.recognition.RecognitionResult;
import com.nizar.venueguard.vision.FaceDetection;
import com.nizar.venueguard.vision.FaceRenderer;
import com.nizar.venueguard.vision.ImagePreprocessor;
import com.nizar.venueguard.vision.PreparedImage;
import com.nizar.venueguard.vision.SFaceEmbedder;
import com.nizar.venueguard.vision.YuNetFaceDetector;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

import java.io.IOException;
import java.nio.file.Path;
import java.sql.SQLException;
import java.time.Duration;
import java.time.Instant;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

import static org.bytedeco.opencv.global.opencv_highgui.WINDOW_AUTOSIZE;
import static org.bytedeco.opencv.global.opencv_highgui.destroyAllWindows;
import static org.bytedeco.opencv.global.opencv_highgui.imshow;
import static org.bytedeco.opencv.global.opencv_highgui.namedWindow;
import static org.bytedeco.opencv.global.opencv_highgui.waitKey;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_SIMPLEX;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_AA;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;

public final class LiveRecognitionRunner {

    private static final int CAMERA_INDEX = 0;

    private static final int REQUESTED_WIDTH = 1280;
    private static final int REQUESTED_HEIGHT = 720;

    private static final int MAXIMUM_IMAGE_DIMENSION = 480;

    private static final Duration EVENT_LOG_INTERVAL =
            Duration.ofSeconds(5);

    /*
     * Recognition is performed once every 12 frames.
     * Face detection still runs on every frame.
     */
    private static final int RECOGNITION_INTERVAL_FRAMES = 12;

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

    private static final String WINDOW_NAME =
            "VenueGuard Live Recognition";

    private LiveRecognitionRunner() {
    }

    public static void main(String[] args)
            throws IOException, SQLException {
        System.out.println(
                "VenueGuard live recognition starting..."
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

        VenueGuardDatabase database =
                new VenueGuardDatabase();

        database.initialize();

        RecognitionEventRepository eventRepository =
                new RecognitionEventRepository(
                        database
                );

        System.out.println(
                "Event database: "
                        + database.databasePath()
        );

        ImagePreprocessor preprocessor =
                new ImagePreprocessor(
                        MAXIMUM_IMAGE_DIMENSION
                );

        try (
                VideoCapture camera =
                        new VideoCapture(CAMERA_INDEX);

                YuNetFaceDetector detector =
                        new YuNetFaceDetector(
                                FACE_DETECTION_MODEL
                        );

                SFaceEmbedder embedder =
                        new SFaceEmbedder(
                                FACE_RECOGNITION_MODEL
                        );

                Mat frame = new Mat()
        ) {
            if (!camera.isOpened()) {
                throw new IllegalStateException(
                        "Could not open camera index "
                                + CAMERA_INDEX
                );
            }

            camera.set(
                    CAP_PROP_FRAME_WIDTH,
                    REQUESTED_WIDTH
            );

            camera.set(
                    CAP_PROP_FRAME_HEIGHT,
                    REQUESTED_HEIGHT
            );

            FaceRecognitionService recognitionService =
                    new FaceRecognitionService(
                            MAXIMUM_IMAGE_DIMENSION,
                            detector,
                            embedder
                    );

            System.out.printf(
                    "Camera opened: %.0f × %.0f%n",
                    camera.get(CAP_PROP_FRAME_WIDTH),
                    camera.get(CAP_PROP_FRAME_HEIGHT)
            );

            System.out.println(
                    "Press Q or Escape in the camera window to stop."
            );

            namedWindow(
                    WINDOW_NAME,
                    WINDOW_AUTOSIZE
            );

            int frameNumber = 0;
            String displayedIdentity = "ANALYSING";
            double displayedSimilarity = Double.NaN;
            boolean displayedRecognition = false;

            Instant lastStoredEventTime = null;
            String lastStoredDecisionKey = null;

            while (true) {
                boolean frameCaptured =
                        camera.read(frame);

                if (!frameCaptured || frame.empty()) {
                    System.err.println(
                            "The camera did not return a valid frame."
                    );

                    break;
                }

                try (
                        PreparedImage prepared =
                                preprocessor.preprocess(frame)
                ) {
                    List<FaceDetection> preparedDetections =
                            detector.detect(
                                    prepared.colour()
                            );

                    float originalScale =
                            (float) (
                                    1.0
                                            / prepared.scaleFactor()
                            );

                    List<FaceDetection> frameDetections =
                            preparedDetections
                                    .stream()
                                    .map(
                                            detection ->
                                                    detection.scaledBy(
                                                            originalScale
                                                    )
                                    )
                                    .toList();

                    FaceDetection primaryFace =
                            findPrimaryFace(
                                    frameDetections
                            );

                    if (primaryFace == null) {
                        displayedIdentity = "NO FACE";
                        displayedSimilarity = Double.NaN;
                        displayedRecognition = false;
                    } else if (
                            frameNumber
                                    % RECOGNITION_INTERVAL_FRAMES
                                    == 0
                    ) {
                        try {
                            RecognitionResult result =
                                    recognitionService.recognize(
                                            frame,
                                            primaryFace,
                                            profile
                                    );

                            displayedIdentity =
                                    result.identity();

                            displayedSimilarity =
                                    result.similarity();

                            displayedRecognition =
                                    result.recognized();

                            Instant eventTime =
                                    Instant.now();

                            String decisionReason =
                                    result.recognized()
                                            ? "SIMILARITY_AT_OR_ABOVE_THRESHOLD"
                                            : "SIMILARITY_BELOW_THRESHOLD";

                            String decisionKey =
                                    result.identity()
                                            + "|"
                                            + result.recognized();

                            boolean decisionChanged =
                                    !decisionKey.equals(
                                            lastStoredDecisionKey
                                    );

                            boolean intervalExpired =
                                    lastStoredEventTime == null
                                            || Duration.between(
                                                    lastStoredEventTime,
                                                    eventTime
                                            )
                                            .compareTo(
                                                    EVENT_LOG_INTERVAL
                                            ) >= 0;

                            if (
                                    decisionChanged
                                            || intervalExpired
                            ) {
                                RecognitionEvent event =
                                        new RecognitionEvent(
                                                eventTime,
                                                result.identity(),
                                                result.recognized(),
                                                result.similarity(),
                                                result.threshold(),
                                                CAMERA_INDEX,
                                                decisionReason
                                        );

                                try {
                                    eventRepository.save(event);

                                    lastStoredEventTime =
                                            eventTime;

                                    lastStoredDecisionKey =
                                            decisionKey;

                                    System.out.printf(
                                            "Stored event: %s, similarity %.4f%n",
                                            result.identity(),
                                            result.similarity()
                                    );
                                } catch (
                                        IOException
                                        | SQLException databaseException
                                ) {
                                    System.err.println(
                                            "Could not store recognition event: "
                                                    + databaseException.getMessage()
                                    );
                                }
                            }
                        } catch (IllegalStateException exception) {
                            displayedIdentity =
                                    "CHECK FACE QUALITY";

                            displayedSimilarity =
                                    Double.NaN;

                            displayedRecognition = false;
                        }
                    }

                    try (
                            Mat renderedFrame =
                                    FaceRenderer.render(
                                            frame,
                                            frameDetections
                                    )
                    ) {
                        if (primaryFace != null) {
                            drawRecognitionLabel(
                                    renderedFrame,
                                    primaryFace,
                                    displayedIdentity,
                                    displayedSimilarity,
                                    displayedRecognition
                            );
                        }

                        imshow(
                                WINDOW_NAME,
                                renderedFrame
                        );
                    }
                }

                frameNumber++;

                if (frameNumber == Integer.MAX_VALUE) {
                    frameNumber = 0;
                }

                int pressedKey =
                        waitKey(1) & 0xFF;

                if (
                        pressedKey == 'q'
                                || pressedKey == 'Q'
                                || pressedKey == 27
                ) {
                    break;
                }
            }

            camera.release();

            System.out.println(
                    "VenueGuard live recognition stopped."
            );
        } finally {
            destroyAllWindows();
        }
    }

    private static FaceDetection findPrimaryFace(
            List<FaceDetection> detections
    ) {
        return detections
                .stream()
                .max(
                        Comparator.comparingDouble(
                                detection ->
                                        (double) detection.width()
                                                * detection.height()
                        )
                )
                .orElse(null);
    }

    private static void drawRecognitionLabel(
            Mat image,
            FaceDetection detection,
            String identity,
            double similarity,
            boolean recognized
    ) {
        String label;

        if (Double.isFinite(similarity)) {
            label =
                    String.format(
                            Locale.ROOT,
                            "%s  %.3f",
                            identity,
                            similarity
                    );
        } else {
            label = identity;
        }

        int labelX =
                Math.max(
                        0,
                        Math.round(detection.x())
                );

        int labelY =
                Math.max(
                        25,
                        Math.round(detection.y()) - 10
                );

        try (
                Point position =
                        new Point(
                                labelX,
                                labelY
                        );

                Scalar colour =
                        recognized
                                ? new Scalar(
                                0,
                                255,
                                0,
                                0
                        )
                                : new Scalar(
                                0,
                                0,
                                255,
                                0
                        )
        ) {
            putText(
                    image,
                    label,
                    position,
                    FONT_HERSHEY_SIMPLEX,
                    0.75,
                    colour,
                    2,
                    LINE_AA,
                    false
            );
        }
    }
}
