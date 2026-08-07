/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/gpu/gpu_diagnostics.h"

#include <iostream>
#include <mutex>

namespace nsparse::detail {

void warn_gpu_fallback_once(const char* context, const char* detail) {
    static std::once_flag flag;
    std::call_once(flag, [&] {
        std::cerr << "nsparse: GPU offload disabled for " << context
                  << ", falling back to CPU (" << detail
                  << "). This message is shown once.\n";
    });
}

}  // namespace nsparse::detail
