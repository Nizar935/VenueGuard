package com.nizar.venueguard.recognition;

import com.nizar.venueguard.enrollment.EnrollmentProfile;
import com.nizar.venueguard.enrollment.FaceTemplate;
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
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

public final class FaceRecognitionService {

    public static final double DEFAULT_COSINE_THRESHOLD =
            0.363;

    private static final int EXPECTED_DIMENSIONS = 128;
    private static final double FACE_CROP_MARGIN = 0.15;

    private final ImagePreprocessor preprocessor;
    private final YuNetFaceDetector detector;
    private final SFaceEmbedder embedder;
    private final FaceQualityAnalyzer qualityAnalyzer;
    private final double similarityThreshold;

    public FaceRecognitionService(
            int maximumImageDimension,
            YuNetFaceDetector detector,
            SFaceEmbedder embedder
    ) {
        this(
                maximumImageDimension,
                detector,
                embedder,
                DEFAULT_COSINE_THRESHOLD
        );
    }

    public FaceRecognitionService(
            int maximumImageDimension,
            YuNetFaceDetector detector,
            SFaceEmbedder embedder,
            double similarityThreshold
    ) {
        if (maximumImageDimension <= 0) {
            throw new IllegalArgumentException(
                    "Maximum image dimension must be positive"
            );
        }

        if (
                !Double.isFinite(similarityThreshold)
                        || similarityThreshold < -1
                        || similarityThreshold > 1
        ) {
            throw new IllegalArgumentException(
                    "Similarity threshold must be "
                            + "between -1 and 1"
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

        this.similarityThreshold =
                similarityThreshold;

        preprocessor =
                new ImagePreprocessor(
                        maximumImageDimension
                );

        qualityAnalyzer =
                new FaceQualityAnalyzer();
    }

    public RecognitionResult recognize(
            Path queryImage,
            EnrollmentProfile profile
    ) throws IOException {
        Objects.requireNonNull(
                queryImage,
                "Query image cannot be null"
        );

        Objects.requireNonNull(
                profile,
                "Enrollment profile cannot be null"
        );

        try (
                Mat original =
                        ImageLoader.loadColour(queryImage);

                PreparedImage prepared =
                        preprocessor.preprocess(original)
        ) {
            List<FaceDetection> detections =
                    detector.detect(
                            prepared.colour()
                    );

            if (detections.isEmpty()) {
                throw new IllegalStateException(
                        "No face was detected in the query image"
                );
            }

            if (detections.size() > 1) {
                throw new IllegalStateException(
                        "Expected one query face but detected "
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
                                "Query face failed quality checks";
                    }

                    throw new IllegalStateException(reason);
                }

                try (
                        Mat detectionRow =
                                originalDetection.toOpenCvRow();

                        FaceEmbedding queryEmbedding =
                                embedder.createEmbedding(
                                        original,
                                        detectionRow
                                )
                ) {
                    return compareAgainstProfile(
                            queryEmbedding.copyValues(),
                            profile
                    );
                }
            }
        }
    }

    private RecognitionResult compareAgainstProfile(
            float[] queryEmbedding,
            EnrollmentProfile profile
    ) {
        validateEmbedding(queryEmbedding);

        double bestSimilarity =
                -Double.MAX_VALUE;

        FaceTemplate bestTemplate = null;

        for (
                FaceTemplate template
                : profile.templates()
        ) {
            float[] enrolledEmbedding =
                    template.embedding();

            double similarity =
                    cosineSimilarity(
                            queryEmbedding,
                            enrolledEmbedding
                    );

            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestTemplate = template;
            }
        }

        if (bestTemplate == null) {
            throw new IllegalStateException(
                    "The profile contains no templates"
            );
        }

        boolean recognized =
                bestSimilarity >= similarityThreshold;

        String identity =
                recognized
                        ? profile.personName()
                        : "UNKNOWN";

        return new RecognitionResult(
                identity,
                recognized,
                bestSimilarity,
                similarityThreshold,
                bestTemplate.sourceImage(),
                profile.sampleCount()
        );
    }

    public RecognitionResult recognize(
            Mat image,
            FaceDetection detection,
            EnrollmentProfile profile
    ) {
        Objects.requireNonNull(
                image,
                "Image cannot be null"
        );

        Objects.requireNonNull(
                detection,
                "Face detection cannot be null"
        );

        Objects.requireNonNull(
                profile,
                "Enrollment profile cannot be null"
        );

        if (image.empty()) {
            throw new IllegalArgumentException(
                    "Image cannot be empty"
            );
        }

        try (
                Mat faceCrop =
                        FaceCropper.crop(
                                image,
                                detection,
                                FACE_CROP_MARGIN
                        )
        ) {
            FaceQualityReport qualityReport =
                    qualityAnalyzer.analyse(faceCrop);

            if (!qualityReport.accepted()) {
                String reason =
                        String.join(
                                "; ",
                                qualityReport.problems()
                        );

                if (reason.isBlank()) {
                    reason =
                            "Live face failed quality checks";
                }

                throw new IllegalStateException(reason);
            }

            try (
                    Mat detectionRow =
                            detection.toOpenCvRow();

                    FaceEmbedding queryEmbedding =
                            embedder.createEmbedding(
                                    image,
                                    detectionRow
                            )
            ) {
                return compareAgainstProfile(
                        queryEmbedding.copyValues(),
                        profile
                );
            }
        }
    }

    private static double cosineSimilarity(
            float[] first,
            float[] second
    ) {
        validateEmbedding(first);
        validateEmbedding(second);

        double dotProduct = 0;
        double firstMagnitudeSquared = 0;
        double secondMagnitudeSquared = 0;

        for (
                int index = 0;
                index < EXPECTED_DIMENSIONS;
                index++
        ) {
            double firstValue = first[index];
            double secondValue = second[index];

            dotProduct +=
                    firstValue * secondValue;

            firstMagnitudeSquared +=
                    firstValue * firstValue;

            secondMagnitudeSquared +=
                    secondValue * secondValue;
        }

        if (
                firstMagnitudeSquared == 0
                        || secondMagnitudeSquared == 0
        ) {
            throw new IllegalArgumentException(
                    "Cannot compare a zero-length embedding"
            );
        }

        return dotProduct
                / (
                Math.sqrt(firstMagnitudeSquared)
                        * Math.sqrt(secondMagnitudeSquared)
        );
    }

    private static void validateEmbedding(
            float[] embedding
    ) {
        Objects.requireNonNull(
                embedding,
                "Embedding cannot be null"
        );

        if (embedding.length != EXPECTED_DIMENSIONS) {
            throw new IllegalArgumentException(
                    "Expected 128 embedding values but received "
                            + embedding.length
            );
        }

        for (float value : embedding) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(
                        "Embedding contains a non-finite value"
                );
            }
        }
    }
}
