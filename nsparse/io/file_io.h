/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef FILE_IO_H
#define FILE_IO_H

#include <cstdio>
#include <string>

#include "nsparse/io/index_io.h"

namespace nsparse {
class FileIOReader : public IOReader {
public:
    explicit FileIOReader(char* file_name);
    explicit FileIOReader(FILE* file);
    ~FileIOReader();

    size_t read(void* ptr, size_t size, size_t nitems) override;
    size_t pos() const override { return pos_; }
    void close() override;

    std::string file_name() { return file_name_; }

private:
    std::string file_name_;
    FILE* file_;
    // Counted rather than ftell()'d: a reader may be handed an already-advanced
    // FILE*, and padding only needs offsets relative to where this reader
    // started.
    size_t pos_ = 0;
};

class FileIOWriter : public IOWriter {
public:
    explicit FileIOWriter(char* filename);
    explicit FileIOWriter(FILE* file);
    ~FileIOWriter();

    void write(void* ptr, size_t size, size_t nitems) override;
    size_t pos() const override { return pos_; }
    void close()
        override;  // Call explicitly if you need error handling on close

private:
    FILE* file_;
    size_t pos_ = 0;
};
}  // namespace nsparse

#endif  // FILE_IO_H