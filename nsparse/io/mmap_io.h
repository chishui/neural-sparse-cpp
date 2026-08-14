/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef MMAP_IO_H
#define MMAP_IO_H

#include "nsparse/io/io.h"
#include "nsparse/utils/mmap_cursor.h"


namespace nsparse {

class MmapSerializable : public Serializable {
public:
    // Borrows the arrays from a mapping rather than copying them: the mapped
    // counterpart of deserialize(), and the reason serialize() pads each array
    // to its element's alignment.
    //
    // `cursor` must sit where serialize() started, and is left just past this
    // object so an enclosing index can go on to read what it wrote next. The
    // mapping must outlive this object, which borrows without owning.
    virtual void mmap_deserialize(MmapCursor* cursor) = 0;
};
}

#endif // MMAP_IO_H