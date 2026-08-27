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

// The fixed-size header written ahead of every index payload, a nested
// delegate's included: fourcc id, format version, dimension. The version comes
// second so that a later change to the rest of the header stays reachable -- id
// and version parse the same way at every version.
//
// Passed around whole rather than as a loose version: the two are adjacent
// integers at every call site that needs them, and a struct is not silently
// swappable with an io_flags or a dimension.
struct IndexHeader {
    uint32_t id = 0;
    // Numbered per index type, not per file: each type versions its own
    // payload, so changing one leaves the others' numbering untouched and a
    // nested delegate carries its own. See IndexIO::format_version.
    uint32_t version = 0;
    int dimension = 0;
};

// Where a payload starts, relative to the header before it. The fields are
// written one at a time with no padding between them.
constexpr size_t kIndexHeaderSize =
    sizeof(uint32_t) + sizeof(uint32_t) + sizeof(int);

class IndexIO {
public:
    virtual ~IndexIO() = default;

    // Layout revision this index type writes. Recorded in the header above and
    // handed back to read_index; bumped only when *this* type's payload
    // changes.
    //
    // Pure rather than defaulted: an inherited version would let a payload
    // change ship without one, which is the failure the header exists to catch.
    [[nodiscard]] virtual uint32_t format_version() const = 0;

    virtual void write_index(IOWriter* io_writer) {};

    // `header` is what the file declared, with its version already checked
    // against format_version(): it is in 1..format_version(), never newer.
    virtual void read_index(IOReader* io_reader, const IndexHeader& header,
                            int io_flags = 0){};
};

constexpr uint32_t fourcc(const std::array<char, 4>& id) {
    return id[0] | id[1] << 8 | id[2] << 16 | id[3] << 24;
}
}  // namespace nsparse

#endif  // IO_H