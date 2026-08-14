/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef BUF_H
#define BUF_H

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <span>

#include "nsparse/io/io.h"

namespace nsparse {

using ReleaseFn = void (*)(void*);

// Sentinel releaser: `MemoryOwner::owns()` compares against this.
inline void release_nothing(void* /*context*/) {}

// Owns a buffer's backing memory without knowing what the buffer holds.
//
// The context is not the data pointer: releasing a mapped array drops a
// reference to the whole mapping rather than freeing the array, so one region
// can back many arrays.
class MemoryOwner {
public:
    MemoryOwner() = default;

    // A null `release` means non-owning, so callers need no borrow special case.
    MemoryOwner(void* context, ReleaseFn release)
        : context_(context, release != nullptr ? release : &release_nothing) {}

    MemoryOwner(const MemoryOwner&) = delete;
    MemoryOwner& operator=(const MemoryOwner&) = delete;
    MemoryOwner(MemoryOwner&&) noexcept = default;
    MemoryOwner& operator=(MemoryOwner&&) noexcept = default;

    [[nodiscard]] bool owns() const {
        return context_.get_deleter() != &release_nothing;
    }

    [[nodiscard]] void* context() const { return context_.get(); }

    [[nodiscard]] ReleaseFn releaser() const { return context_.get_deleter(); }

    // nullptr unless `expected` is the releaser held, so the releaser identity
    // serves as a type tag for the context.
    template <class C>
    [[nodiscard]] C* context_as(ReleaseFn expected) const {
        return releaser() == expected ? static_cast<C*>(context()) : nullptr;
    }

    // Gives up ownership without releasing; the caller inherits the context.
    [[nodiscard]] void* release() { return context_.release(); }

private:
    // A function-pointer deleter keeps an owner two words and allocation-free.
    std::unique_ptr<void, ReleaseFn> context_{nullptr, &release_nothing};
};

// A fixed-length, contiguous, read-only array of T that either owns its memory
// or borrows it. One access path either way, so in-memory and mapped indexes run
// the same code.
//
// Fixed-length because a mapping cannot resize: growth belongs to the build
// path, which appends into a std::vector and hands it over via own().
template <class T>
class Buf {
    static_assert(std::is_trivially_copyable_v<T>,
                  "Buf borrows bytes in place, so T must be memcpy-able");

public:
    using value_type = T;

    Buf() = default;

    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&&) noexcept = default;
    Buf& operator=(Buf&&) noexcept = default;

    // Adopts `vector`'s elements without copying: the vector object becomes the
    // release context, so its buffer stays put.
    static Buf own(std::vector<T>&& vector) {
        const size_t size = vector.size();
        auto held = std::make_unique<std::vector<T>>(std::move(vector));
        T* data = held->data();
        return Buf(data, size, held.release(), &release_vector);
    }

    // Points at memory the caller keeps alive: a mapping, another Buf's
    // interior, or a query array owned by search()'s caller.
    static Buf borrow(const T* data, size_t size) {
        return Buf(data, size, nullptr, nullptr);
    }

    // Points at `data` while owning `context`. Since the two are independent,
    // several arrays can each hold a reference to the region they came from.
    static Buf adopt(const T* data, size_t size, void* context,
                     ReleaseFn release) {
        return Buf(data, size, context, release);
    }

    [[nodiscard]] const T* data() const { return data_; }

    [[nodiscard]] size_t size() const { return size_; }

    [[nodiscard]] size_t byte_size() const { return size_ * sizeof(T); }

    [[nodiscard]] bool empty() const { return size_ == 0; }

    const T& operator[](size_t index) const { return data_[index]; }

    [[nodiscard]] const T& at(size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("Buf index out of range");
        }
        return data_[index];
    }

    [[nodiscard]] const T* begin() const { return data_; }
    [[nodiscard]] const T* end() const { return data_ + size_; }

    [[nodiscard]] bool owns() const { return owner_.owns(); }

    // The vector handed to own(), or nullptr for any other origin.
    [[nodiscard]] const std::vector<T>* owned_vector() const {
        return owner_.template context_as<std::vector<T>>(&release_vector);
    }

    // Moves the owned vector out, leaving this Buf empty. An append grows a Buf
    // by taking the vector, extending it, and own()ing it again; capacity
    // survives the round trip, keeping appends amortized.
    //
    // A default-constructed Buf yields an empty vector, so a first append needs
    // no special case. Borrowed and adopted buffers throw: growing them would
    // mean writing to memory this Buf does not own.
    [[nodiscard]] std::vector<T> take_vector() {
        if (!owner_.owns() && data_ == nullptr) {
            return {};
        }
        auto* owned = owner_.template context_as<std::vector<T>>(
            &release_vector);
        if (owned == nullptr) {
            throw std::runtime_error(
                "Buf does not own a vector: borrowed and mapped buffers cannot "
                "grow");
        }
        std::vector<T> taken = std::move(*owned);
        // Frees the now-empty vector object left behind by the move.
        owner_ = MemoryOwner();
        data_ = nullptr;
        size_ = 0;
        return taken;
    }

    std::span<const T> span() {return std::span<const T>(data_, size_);}

private:
    Buf(const T* data, size_t size, void* context, ReleaseFn release)
        : data_(data), size_(size), owner_(context, release) {}

    static void release_vector(void* context) {
        delete static_cast<std::vector<T>*>(context);
    }

    // Typed and outside the owner: only the context needs erasing, so reads
    // take no cast.
    const T* data_ = nullptr;
    size_t size_ = 0;
    MemoryOwner owner_;
};

// Reads `count` T from `reader` into an owned Buf. Zero is not a short read:
// an empty array is a legitimate encoding, and reading nothing keeps the stream
// where the writer left it.
template <class T>
Buf<T> read_owned(IOReader* reader, size_t count) {
    if (count == 0) {
        return {};
    }
    std::vector<T> values(count);
    if (reader->read(values.data(), sizeof(T), count) != count) {
        throw std::runtime_error("Truncated array in index file");
    }
    return Buf<T>::own(std::move(values));
}

}  // namespace nsparse

#endif  // BUF_H
