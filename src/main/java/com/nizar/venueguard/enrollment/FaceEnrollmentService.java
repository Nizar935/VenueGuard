package com.nizar.venueguard.enrollment;

import com.nizar.venueguard.vision.FaceCropper;
import com.nizar.venueguard.vision.FaceDetection;
import com.nizar.venueguard.vision.FaceEmbedding;
import com.nizar.venueguard.vision.FaceQualityAnalyzer;
import com.nizar.venueguard.vision.FaceQualityReport;
import com.nizar.venueguard.vision.ImageLoader;
import com.nizar.venueguard.vision.ImagePreprocessor;
import com.nizar.venueguard.vision.PreparedImage;
import com.nizar.venueguard.vision.SFaceEmbedder;
import com.nizar.venueguard.vision.YuNetFaceDetector;
import org.bytedeco.opencv.opencv_core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.stream.Stream;

public final class FaceEnrollmentService {

    private static final double FACE_CROP_MARGIN = 0.15;

    private final ImagePreprocessor preprocessor;
    private final YuNetFaceDetector detector;
    private final SFaceEmbedder embedder;
    private final FaceQualityAnalyzer qualityAnalyzer;

    public FaceEnrollmentService(
            int maximumImageDimension,
            YuNetFaceDetector detector,
            SFaceEmbedder embedder
    ) {
        if (maximumImageDimension <= 0) {
            throw new IllegalArgumentException(
                    "Maximum image dimension must be positive"
            );
        }

        this.detector = Objects.requireNonNull(
                detector,
                "Face detector cannot be null"
        );

        this.embedder = Objects.requireNonNull(
                embedder,
                "Face embedder cannot be null"
        );

        preprocessor =
                new ImagePreprocessor(
                        maximumImageDimension
                );

        qualityAnalyzer =
                new FaceQualityAnalyzer();
    }

    public EnrollmentReport enroll(
            Path personDirectory
    ) throws IOException {
        Objects.requireNonNull(
                personDirectory,
                "Person directory cannot be null"
        );

        Path absoluteDirectory =
                personDirectory.toAbsolutePath().normalize();

        if (!Files.isDirectory(absoluteDirectory)) {
            throw new IllegalArgumentException(
                    "Enrollment directory does not exist: "
                            + absoluteDirectory
            );
        }

        Path directoryName =
                absoluteDirectory.getFileName();

        if (directoryName == null) {
            throw new IllegalArgumentException(
                    "Could not determine the person's name "
                            + "from the directory"
            );
        }

        String personName =
                directoryName.toString();

        List<Path> imagePaths =
                findImageFiles(absoluteDirectory);

        if (imagePaths.isEmpty()) {
            throw new IllegalArgumentException(
                    "No supported photographs were found in: "
                            + absoluteDirectory
            );
        }

        List<FaceTemplate> acceptedTemplates =
                new ArrayList<>();

        List<RejectedEnrollmentImage> rejectedImages =
                new ArrayList<>();

        for (Path imagePath : imagePaths) {
            ImageProcessingResult result;

            try {
                result = processImage(imagePath);
            } catch (IOException | RuntimeException exception) {
                String message = exception.getMessage();

                if (message == null || message.isBlank()) {
                    message =
                            exception.getClass().getSimpleName();
                }

                result = ImageProcessingResult.rejected(
                        "Could not process image: " + message
                );
            }

            if (result.wasAccepted()) {
                acceptedTemplates.add(
                        result.template()
                );
            } else {
                rejectedImages.add(
                        new RejectedEnrollmentImage(
                                imagePath,
                                result.rejectionReason()
                        )
                );
            }
        }

        if (acceptedTemplates.isEmpty()) {
            throw new IllegalStateException(
                    "Enrollment failed because none of the "
                            + imagePaths.size()
                            + " photographs passed validation"
            );
        }

        EnrollmentProfile profile =
                new EnrollmentProfile(
                        personName,
                        acceptedTemplates
                );

        return new EnrollmentReport(
                profile,
                rejectedImages
        );
    }

    private ImageProcessingResult processImage(
            Path imagePath
    ) throws IOException {
        try (
                Mat original =
                        ImageLoader.loadColour(imagePath);

                PreparedImage prepared =
                        preprocessor.preprocess(original)
        ) {
            List<FaceDetection> detections =
                    detector.detect(
                            prepared.colour()
                    );

            if (detections.isEmpty()) {
                return ImageProcessingResult.rejected(
                        "No face was detected"
                );
            }

            if (detections.size() > 1) {
                return ImageProcessingResult.rejected(
                        "Expected one face but detected "
                                + detections.size()
                );
            }

            float originalScale =
                    (float) (
                            1.0
                                    / prepared.scaleFactor()
                    );

            FaceDetection originalDetection =
                    detections
                            .get(0)
                            .scaledBy(originalScale);

            try (
                    Mat faceCrop =
                            FaceCropper.crop(
                                    original,
                                    originalDetection,
                                    FACE_CROP_MARGIN
                            )
            ) {
                FaceQualityReport qualityReport =
                        qualityAnalyzer.analyse(
                                faceCrop
                        );

                if (!qualityReport.accepted()) {
                    String reason =
                            String.join(
                                    "; ",
                                    qualityReport.problems()
                            );

                    if (reason.isBlank()) {
                        reason =
                                "The face failed quality checks";
                    }

                    return ImageProcessingResult.rejected(
                            reason
                    );
                }

                try (
                        Mat detectionRow =
                                originalDetection.toOpenCvRow();

                        FaceEmbedding embedding =
                                embedder.createEmbedding(
                                        original,
                                        detectionRow
                                )
                ) {
                    FaceTemplate template =
                            new FaceTemplate(
                                    imagePath,
                                    embedding.copyValues()
                            );

                    return ImageProcessingResult.accepted(
                            template
                    );
                }
            }
        }
    }

    private static List<Path> findImageFiles(
            Path directory
    ) throws IOException {
        try (
                Stream<Path> files =
                        Files.list(directory)
        ) {
            return files
                    .filter(Files::isRegularFile)
                    .filter(
                            FaceEnrollmentService
                                    ::isSupportedImage
                    )
                    .sorted(
                            Comparator.comparing(
                                    (Path path) -> path
                                            .getFileName()
                                            .toString()
                                            .toLowerCase(
                                                    Locale.ROOT
                                            )
                            )
                    )
                    .toList();
        }
    }

    private static boolean isSupportedImage(
            Path path
    ) {
        String filename =
                path.getFileName()
                        .toString()
                        .toLowerCase(Locale.ROOT);

        return filename.endsWith(".jpg")
                || filename.endsWith(".jpeg")
                || filename.endsWith(".png")
                || filename.endsWith(".bmp");
    }

    private record ImageProcessingResult(
            FaceTemplate template,
            String rejectionReason
    ) {

        private static ImageProcessingResult accepted(
                FaceTemplate template
        ) {
            return new ImageProcessingResult(
                    Objects.requireNonNull(
                            template,
                            "Accepted template cannot be null"
                    ),
                    null
            );
        }

        private static ImageProcessingResult rejected(
                String reason
        ) {
            return new ImageProcessingResult(
                    null,
                    Objects.requireNonNull(
                            reason,
                            "Rejection reason cannot be null"
                    )
            );
        }

        private boolean wasAccepted() {
            return template != null;
        }
    }
}
