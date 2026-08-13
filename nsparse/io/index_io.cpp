/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/io/index_io.h"

#include <cstdint>
#include <functional>
#include <stdexcept>

#include "nsparse/brutal_index.h"
#include "nsparse/id_map_index.h"
#include "nsparse/inverted_index.h"
#include "nsparse/io/file_io.h"
#include "nsparse/seismic_index.h"
#include "nsparse/seismic_scalar_quantized_index.h"

namespace nsparse {

namespace {
constexpr uint32_t BRUT = fourcc(BrutalIndex::name);
constexpr uint32_t SEIS = fourcc(SeismicIndex::name);
constexpr uint32_t SESQ = fourcc(SeismicScalarQuantizedIndex::name);
constexpr uint32_t IDMP = fourcc(IDMapIndex::name);
constexpr uint32_t INVT = fourcc(InvertedIndex::name);

// Closed here rather than in a scope guard: close() reports flush/fclose
// failures by throwing, which a destructor cannot forward.
template <class T>
void close_stream(T* stream, bool keep_open) {
    if (!keep_open) {
        stream->close();
    }
}

void write_header(Index* index, IOWriter* io_writer) {
    // write index type
    auto id_val = fourcc(index->id());
    io_writer->write(&id_val, sizeof(uint32_t), 1);
    // write dimension
    auto dimension = index->get_dimension();
    io_writer->write(&dimension, sizeof(int), 1);
}

Index* read_header(IOReader* io_reader) {
    uint32_t id_val = 0;
    io_reader->read(&id_val, sizeof(uint32_t), 1);
    int dimension = 0;
    io_reader->read(&dimension, sizeof(int), 1);
    switch (id_val) {
        case BRUT:
            return new BrutalIndex(dimension);
        case SEIS:
            return new SeismicIndex(dimension);
        case SESQ:
            return new SeismicScalarQuantizedIndex(dimension);
        case IDMP:
            return new IDMapIndex();
        case INVT:
            return new InvertedIndex(dimension);
        default:
            throw std::runtime_error("Unknown index type");
    }
}
}  // namespace

namespace detail {
void write_index(Index* index, IOWriter* io_writer, bool keep_open) {
    auto* index_io = dynamic_cast<IndexIO*>(index);
    auto auto_close = [keep_open](auto* stream) { close_stream(stream, keep_open); };
    auto io_writer_ptr = std::unique_ptr<IOWriter, decltype(auto_close)>(io_writer, auto_close);
    if (index_io == nullptr) {
        throw std::runtime_error("Index does not support serialization");
    }
    // write header
    write_header(index, io_writer_ptr.get());
    // write index customized payload
    index_io->write_index(io_writer_ptr.get());
}

Index* read_index(IOReader* io_reader, bool keep_open, int io_flags) {
    auto auto_close = [keep_open](auto* stream) { close_stream(stream, keep_open); };
    std::unique_ptr<IOReader, decltype(auto_close)> io_reader_ptr(io_reader, auto_close);
    Index* index = read_header(io_reader_ptr.get());
    auto* index_io = dynamic_cast<IndexIO*>(index);
    if (index_io == nullptr) {
        throw std::runtime_error("Index does not support serialization");
    }

    // handle mmap
    if (fourcc(index->id()) == SEIS &&
        (io_flags & IndexIoFlag::kUseMmap) == IndexIoFlag::kUseMmap) {
        if (auto* file_io_reader = dynamic_cast<FileIOReader*>(io_reader)) {
            int dimension = index->get_dimension();
            // Where the payload starts, which is what serialize() padded against.
            size_t pos = io_reader_ptr->pos();
            delete index;
            io_reader_ptr.reset();
            return SeismicIndex::mmap_index(dimension, file_io_reader->file_name().c_str(), pos);
        }
    }

    index_io->read_index(io_reader_ptr.get(), io_flags);
    return index;
}
}  // namespace detail

void write_index(Index* index, IOWriter* io_writer) {
    detail::write_index(index, io_writer, false);
}

void write_index(Index* index, char* file_name) {
    FileIOWriter writer(file_name);
    write_index(index, &writer);
}

Index* read_index(IOReader* io_reader, int io_flags) {
    return detail::read_index(io_reader, false, io_flags);
}

Index* read_index(char* file_name, int io_flags) {
    FileIOReader reader(file_name);
    return read_index(&reader, io_flags);
}
}  // namespace nsparse
