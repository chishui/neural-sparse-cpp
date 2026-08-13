/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef IO_ALIGN_H
#define IO_ALIGN_H

#include <array>
#include <cstddef>
#include <cstdint>

#include "nsparse/io/io.h"
#include "nsparse/utils/buf.h"

// Alignment padding for serialized arrays.
//
// Nothing about a stream of writes lands each array on a boundary its element
// type can be dereferenced from -- a term_t array of odd length leaves the next
// float array 2 bytes off. So a serializer pads before writing each array and a
// deserializer skips the same padding before reading it, at a cost of under 8
// bytes per array.
//
// Today's readers copy, and a copy is aligned wherever it came from. The padding
// is for the reader that does not: borrowing an array in place needs it aligned
// in the file, since misaligned loads are UB on x86 and fault on ARM. Writing it
// now keeps that option open without a second on-disk format.
namespace nsparse::io_align {

// Largest alignment any serialized element needs, and so the most padding a
// single call can insert.
constexpr size_t kMaxAlignment = 8;

// Bytes to insert at `pos` to reach an `alignment` boundary.
constexpr size_t padding_for(size_t pos, size_t alignment) {
    return (alignment - pos % alignment) % alignment;
}

// Pads the stream so the next write starts on an `alignment` boundary.
inline void pad_to(IOWriter* writer, size_t alignment) {
    const size_t bytes = padding_for(writer->pos(), alignment);
    if (bytes > 0) {
        std::array<uint8_t, kMaxAlignment> zeros{};
        writer->write(zeros.data(), 1, bytes);
    }
}

// Consumes the padding pad_to() wrote at the same point in the layout.
inline void skip_padding(IOReader* reader, size_t alignment) {
    const size_t bytes = padding_for(reader->pos(), alignment);
    if (bytes > 0) {
        std::array<uint8_t, kMaxAlignment> discard{};
        reader->read(discard.data(), 1, bytes);
    }
}

// read_owned, preceded by the padding pad_to() wrote. `alignment` defaults to
// T's, but a byte array that is reinterpreted at a wider element size on read
// needs that width passed instead.
template <class T>
Buf<T> read_padded(IOReader* reader, size_t count,
                   size_t alignment = alignof(T)) {
    if (count == 0) {
        return {};
    }
    skip_padding(reader, alignment);
    return read_owned<T>(reader, count);
}

}  // namespace nsparse::io_align

#endif  // IO_ALIGN_H
