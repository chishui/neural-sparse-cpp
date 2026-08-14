/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef IO_H
#define IO_H
#include <array>
#include <cstddef>
#include <cstdint>


namespace nsparse {
class IOReader {
public:
    virtual ~IOReader() = default;
    virtual size_t read(void* ptr, size_t size, size_t nitems) = 0;

    // Bytes consumed so far, counted from the start of the stream. Padding is
    // computed from the absolute offset, so this has to agree with the writer's
    // pos() at the same point in the layout.
    [[nodiscard]] virtual size_t pos() const = 0;

    virtual void close() {}
};

class IOWriter {
public:
    virtual ~IOWriter() = default;
    virtual void write(void* ptr, size_t size, size_t nitems) = 0;

    // Bytes written so far. Serializers pad against it so the arrays that
    // follow land on their natural boundary and can be mapped in place;
    // without it a writer would have to guess the offset it is writing at,
    // which no amount of validation downstream could recover.
    [[nodiscard]] virtual size_t pos() const = 0;

    virtual void close() {}
};

class Serializable {
public:
    virtual ~Serializable() = default;
    virtual void serialize(IOWriter* writer) const = 0;
    virtual void deserialize(IOReader* reader) = 0;
};


class IndexIO {
public:
    virtual ~IndexIO() = default;
    virtual void write_index(IOWriter* io_writer) {};
    virtual void read_index(IOReader* io_reader, int io_flags = 0) {};
};

constexpr uint32_t fourcc(const std::array<char, 4>& id) {
    return id[0] | id[1] << 8 | id[2] << 16 | id[3] << 24;
}
}  // namespace nsparse

#endif  // IO_H