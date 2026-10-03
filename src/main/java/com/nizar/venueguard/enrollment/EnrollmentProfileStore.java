package com.nizar.venueguard.enrollment;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class EnrollmentProfileStore {

    private static final int FILE_MAGIC = 0x56475031;
    private static final int FILE_VERSION = 1;

    private static final int EMBEDDING_DIMENSIONS = 128;
    private static final int MAXIMUM_TEMPLATE_COUNT = 1_000;

    public void save(
            EnrollmentProfile profile,
            Path profilePath
    ) throws IOException {
        Objects.requireNonNull(
                profile,
                "Enrollment profile cannot be null"
        );

        Objects.requireNonNull(
                profilePath,
                "Profile path cannot be null"
        );

        Path absoluteProfilePath =
                profilePath.toAbsolutePath().normalize();

        Path parentDirectory =
                absoluteProfilePath.getParent();

        if (parentDirectory != null) {
            Files.createDirectories(parentDirectory);
        }

        Path temporaryPath =
                absoluteProfilePath.resolveSibling(
                        absoluteProfilePath
                                .getFileName()
                                + ".tmp"
                );

        try {
            writeProfile(
                    profile,
                    temporaryPath
            );

            moveIntoPlace(
                    temporaryPath,
                    absoluteProfilePath
            );
        } finally {
            Files.deleteIfExists(temporaryPath);
        }
    }

    public EnrollmentProfile load(
            Path profilePath
    ) throws IOException {
        Objects.requireNonNull(
                profilePath,
                "Profile path cannot be null"
        );

        Path absoluteProfilePath =
                profilePath.toAbsolutePath().normalize();

        if (!Files.isRegularFile(absoluteProfilePath)) {
            throw new IOException(
                    "Enrollment profile does not exist: "
                            + absoluteProfilePath
            );
        }

        try (
                DataInputStream input =
                        new DataInputStream(
                                new BufferedInputStream(
                                        Files.newInputStream(
                                                absoluteProfilePath
                                        )
                                )
                        )
        ) {
            int fileMagic = input.readInt();

            if (fileMagic != FILE_MAGIC) {
                throw new IOException(
                        "The file is not a VenueGuard profile"
                );
            }

            int version = input.readInt();

            if (version != FILE_VERSION) {
                throw new IOException(
                        "Unsupported profile version: "
                                + version
                );
            }

            String personName =
                    input.readUTF();

            int templateCount =
                    input.readInt();

            if (
                    templateCount <= 0
                            || templateCount
                            > MAXIMUM_TEMPLATE_COUNT
            ) {
                throw new IOException(
                        "Invalid template count: "
                                + templateCount
                );
            }

            List<FaceTemplate> templates =
                    new ArrayList<>(templateCount);

            for (
                    int templateIndex = 0;
                    templateIndex < templateCount;
                    templateIndex++
            ) {
                String sourceFilename =
                        input.readUTF();

                int dimensions =
                        input.readInt();

                if (dimensions != EMBEDDING_DIMENSIONS) {
                    throw new IOException(
                            "Invalid embedding dimensions: "
                                    + dimensions
                    );
                }

                float[] embedding =
                        new float[dimensions];

                for (
                        int valueIndex = 0;
                        valueIndex < dimensions;
                        valueIndex++
                ) {
                    float value =
                            input.readFloat();

                    if (!Float.isFinite(value)) {
                        throw new IOException(
                                "Profile contains an invalid "
                                        + "embedding value"
                        );
                    }

                    embedding[valueIndex] = value;
                }

                Path sourcePath =
                        Path.of(
                                "data",
                                "enrollment",
                                personName,
                                sourceFilename
                        );

                templates.add(
                        new FaceTemplate(
                                sourcePath,
                                embedding
                        )
                );
            }

            return new EnrollmentProfile(
                    personName,
                    templates
            );
        }
    }

    private static void writeProfile(
            EnrollmentProfile profile,
            Path outputPath
    ) throws IOException {
        try (
                DataOutputStream output =
                        new DataOutputStream(
                                new BufferedOutputStream(
                                        Files.newOutputStream(
                                                outputPath,
                                                StandardOpenOption.CREATE,
                                                StandardOpenOption
                                                        .TRUNCATE_EXISTING,
                                                StandardOpenOption.WRITE
                                        )
                                )
                        )
        ) {
            output.writeInt(FILE_MAGIC);
            output.writeInt(FILE_VERSION);

            output.writeUTF(
                    profile.personName()
            );

            output.writeInt(
                    profile.sampleCount()
            );

            for (
                    FaceTemplate template
                    : profile.templates()
            ) {
                Path filename =
                        template.sourceImage()
                                .getFileName();

                if (filename == null) {
                    throw new IOException(
                            "Template has no source filename"
                    );
                }

                output.writeUTF(
                        filename.toString()
                );

                float[] embedding =
                        template.embedding();

                if (
                        embedding.length
                                != EMBEDDING_DIMENSIONS
                ) {
                    throw new IOException(
                            "Expected a 128-value embedding"
                    );
                }

                output.writeInt(
                        embedding.length
                );

                for (float value : embedding) {
                    output.writeFloat(value);
                }
            }
        }
    }

    private static void moveIntoPlace(
            Path temporaryPath,
            Path profilePath
    ) throws IOException {
        try {
            Files.move(
                    temporaryPath,
                    profilePath,
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING
            );
        } catch (
                AtomicMoveNotSupportedException exception
        ) {
            Files.move(
                    temporaryPath,
                    profilePath,
                    StandardCopyOption.REPLACE_EXISTING
            );
        }
    }
}