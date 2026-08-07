/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef GPU_DIAGNOSTICS_H
#define GPU_DIAGNOSTICS_H

namespace nsparse::detail {

// Warn once per process that GPU offload for `context` failed and the build is
// falling back to CPU. A systematic failure (wrong arch, cuSPARSE mismatch,
// OOM) otherwise runs the whole build on CPU with no signal.
void warn_gpu_fallback_once(const char* context, const char* detail);

}  // namespace nsparse::detail

#endif  // GPU_DIAGNOSTICS_H
