/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef OMP_EXCEPTION_H
#define OMP_EXCEPTION_H
#include <atomic>
#include <exception>
#include <mutex>

namespace nsparse::detail {

// An exception leaving an OpenMP region is std::terminate, so SWIG/JNI callers
// would see an abort instead of an error. Run each iteration of a parallel loop
// body through run(): the first exception is kept and the remaining iterations
// are skipped (they cannot break out of an `omp for`), then rethrow() raises it
// once the region has ended.
class OmpExceptionCapture {
public:
    template <class Body>
    void run(Body&& body) noexcept {
        if (failed_.load(std::memory_order_relaxed)) {
            return;
        }
        try {
            body();
        } catch (...) {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) {
                error_ = std::current_exception();
            }
            failed_.store(true, std::memory_order_relaxed);
        }
    }

    void rethrow() const {
        if (error_) {
            std::rethrow_exception(error_);
        }
    }

private:
    std::atomic<bool> failed_{false};
    std::mutex mutex_;
    std::exception_ptr error_;
};

}  // namespace nsparse::detail

#endif  // OMP_EXCEPTION_H
