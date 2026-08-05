/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/cluster/inverted_list_clusters.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#ifdef NSPARSE_WITH_GPU
#include "nsparse/gpu/gpu_summarizer.h"
#endif

namespace nsparse {
namespace {

// Prune a cluster's (term, max-value) pairs at the alpha mass cutoff and append
// the survivors (in ascending term order) as a summary vector. Shared by the
// CPU and GPU max-pool paths so their output is identical.
template <class T>
void summarize_emit_cluster_(std::vector<std::pair<term_t, T>>& pairs,
                             float sum, float alpha, SparseVectors& out) {
    std::ranges::sort(pairs, [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    float addup = 0.0F;
    for (size_t j = 0; j < pairs.size(); ++j) {
        addup += pairs[j].second;
        if (addup / sum >= alpha) {
            pairs.erase(pairs.begin() + j + 1, pairs.end());
            break;
        }
    }
    std::ranges::sort(pairs, [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    std::vector<term_t> terms;
    std::vector<T> values;
    terms.reserve(pairs.size());
    values.reserve(pairs.size());
    for (const auto& [term, value] : pairs) {
        terms.push_back(term);
        values.push_back(value);
    }
    out.add_vector(terms.data(), terms.size(),
                   reinterpret_cast<const uint8_t*>(values.data()),
                   values.size() * sizeof(T));
}

// CPU per-term max-pool over a flat, dim-sized accumulator reused across all
// clusters (replaces a per-cluster std::unordered_map, ~2.8x cheaper for the
// many small clusters). A per-cluster epoch marks live slots, so the buffers
// need no per-cluster reset; first touch is detected via the epoch (not
// acc==0), robust to zero-valued weights.
template <class T>
SparseVectors summarize_with_cpu_(const SparseVectors* vectors,
                                  const std::vector<idx_t>& group_of_doc_ids,
                                  const std::vector<idx_t>& offsets,
                                  float alpha) {
    SparseVectors out({.element_size = vectors->get_element_size(),
                       .dimension = vectors->get_dimension()});
    const size_t n_clusters = offsets.size() - 1;
    const auto& indptr_data = vectors->indptr_data();
    const auto& indices_data = vectors->indices_data();
    const auto& values_data = vectors->values_data();
    const size_t dim = vectors->get_dimension();

    std::vector<T> acc(dim, T(0));
    std::vector<uint32_t> epoch(dim, 0);
    std::vector<term_t> touched;
    uint32_t cur_epoch = 0;

    for (size_t i = 0; i < n_clusters; ++i) {
        ++cur_epoch;
        touched.clear();
        float sum = 0.0F;
        auto doc_ids = std::span<const idx_t>(
            group_of_doc_ids.data() + offsets[i], offsets[i + 1] - offsets[i]);
        for (const auto& doc_id : doc_ids) {
            int start = indptr_data[doc_id];
            int end = indptr_data[doc_id + 1];
            for (size_t j = start; j < end; ++j) {
                const term_t term = indices_data[j];
                // j is element index, need byte offset for T access
                const T v =
                    *reinterpret_cast<const T*>(values_data + j * sizeof(T));
                if (epoch[term] != cur_epoch) {
                    // First occurrence in this cluster; value = max(0, v).
                    epoch[term] = cur_epoch;
                    const T value = std::max(T(0), v);
                    acc[term] = value;
                    touched.push_back(term);
                    sum += value;
                } else if (v > acc[term]) {
                    sum += v - acc[term];
                    acc[term] = v;
                }
            }
        }
        std::vector<std::pair<term_t, T>> pairs;
        pairs.reserve(touched.size());
        for (const term_t term : touched) {
            pairs.emplace_back(term, acc[term]);
        }
        summarize_emit_cluster_<T>(pairs, sum, alpha, out);
    }
    return out;
}

#ifdef NSPARSE_WITH_GPU
// GPU per-term max-pool: one kernel launch for the whole list, then the shared
// CPU sort/truncate. float (U32) only. Returns std::nullopt when the GPU
// declines (unavailable / empty list) so the caller falls back to the CPU path.
template <class T>
std::optional<SparseVectors> summarize_with_gpu_(
    const SparseVectors* vectors, const std::vector<idx_t>& group_of_doc_ids,
    const std::vector<idx_t>& offsets, float alpha) {
    if constexpr (!std::is_same_v<T, float>) {
        return std::nullopt;  // GPU max-pool is float-only
    } else {
        const size_t n_clusters = offsets.size() - 1;
        std::vector<detail::GpuSummarizer::ClusterSummary> gpu_clusters;
        if (!detail::GpuSummarizer::instance().summarize_list(
                vectors, group_of_doc_ids.data(), offsets.data(), n_clusters,
                gpu_clusters)) {
            return std::nullopt;
        }
        SparseVectors out({.element_size = vectors->get_element_size(),
                           .dimension = vectors->get_dimension()});
        for (size_t i = 0; i < n_clusters; ++i) {
            const auto& gc = gpu_clusters[i];
            std::vector<std::pair<term_t, T>> pairs;
            pairs.reserve(gc.terms.size());
            for (size_t t = 0; t < gc.terms.size(); ++t) {
                pairs.emplace_back(gc.terms[t], gc.values[t]);
            }
            summarize_emit_cluster_<T>(pairs, gc.sum, alpha, out);
        }
        return out;
    }
}
#endif

/**
 * @brief Generate the per-cluster summary sparse vector (CSR) for a posting
 *        list's clusters.
 *
 * @param vectors inverted index
 * @param group_of_doc_ids flattened doc ids of all clusters
 * @param offsets cluster boundaries into group_of_doc_ids
 * @param alpha prune ratio
 * @return SparseVectors  one summary vector per cluster
 */
template <class T>
SparseVectors summarize_(const SparseVectors* vectors,
                         const std::vector<idx_t>& group_of_doc_ids,
                         const std::vector<idx_t>& offsets, float alpha) {
    if (offsets.size() <= 1) {
        return SparseVectors({.element_size = vectors->get_element_size(),
                              .dimension = vectors->get_dimension()});
    }
#ifdef NSPARSE_WITH_GPU
    // GPU max-pool is opt-in (NSPARSE_GPU_SUMMARIZE=1); fall back to CPU when
    // disabled, unsupported for T, or the GPU declines.
    if (detail::should_offload_summarize_to_gpu()) {
        if (auto gpu_result = summarize_with_gpu_<T>(vectors, group_of_doc_ids,
                                                     offsets, alpha)) {
            return std::move(*gpu_result);
        }
    }
#endif
    return summarize_with_cpu_<T>(vectors, group_of_doc_ids, offsets, alpha);
}

}  // namespace

InvertedListClusters::InvertedListClusters(
    const std::vector<std::vector<idx_t>>& docs) {
    if (docs.empty()) return;
    offsets_.reserve(docs.size() + 1);
    offsets_.push_back(0);
    for (const auto& doc_ids : docs) {
        docs_.insert(docs_.end(), doc_ids.begin(), doc_ids.end());
        offsets_.push_back(docs_.size());
    }
}

InvertedListClusters::InvertedListClusters(const InvertedListClusters& other) =
    default;
InvertedListClusters& InvertedListClusters::operator=(
    const InvertedListClusters& other) = default;

auto InvertedListClusters::get_docs(idx_t idx) const -> std::span<const idx_t> {
    return {docs_.data() + offsets_[idx],
            static_cast<size_t>(offsets_[idx + 1] - offsets_[idx])};
}

void InvertedListClusters::summarize(const SparseVectors* vectors,
                                     float alpha) {
    const auto element_size = vectors->get_element_size();
    SparseVectors summaries;
    if (element_size == U32) {
        summaries = summarize_<float>(vectors, docs_, offsets_, alpha);
    } else if (element_size == U16) {
        summaries = summarize_<uint16_t>(vectors, docs_, offsets_, alpha);
    } else {
        summaries = summarize_<uint8_t>(vectors, docs_, offsets_, alpha);
    }
    build_transpose(summaries);
}

// Build the term-major (CSC) transpose directly from a per-cluster CSR summary.
// The CSR summary is transient; only the transpose is retained.
void InvertedListClusters::build_transpose(const SparseVectors& summaries) {
    n_clusters_ = summaries.num_vectors();
    element_size_ = summaries.get_element_size();
    term_ids_.clear();
    term_ptr_.clear();
    csc_cluster_.clear();
    csc_value_.clear();
    if (n_clusters_ == 0) {
        return;
    }
    // cluster_id_t (uint16) must be able to represent every cluster index in
    // this list; beta keeps this well under 2^16 for any workable config.
    assert(n_clusters_ <= std::numeric_limits<cluster_id_t>::max() + size_t{1});

    const auto* indptr = summaries.indptr_data();
    const auto* indices = summaries.indices_data();
    const auto* values = summaries.values_data();  // raw bytes
    const size_t nnz = static_cast<size_t>(indptr[n_clusters_]);
    const size_t esz = element_size_;

    // Distinct summary terms, ascending. Collected by sort+unique over the
    // summary indices so no dimension-sized scratch is needed (the working set
    // stays proportional to nnz).
    term_ids_.assign(indices, indices + nnz);
    std::ranges::sort(term_ids_);
    term_ids_.erase(std::ranges::unique(term_ids_).begin(), term_ids_.end());
    const size_t n_terms = term_ids_.size();

    // Maps a term to its index in term_ids_ (its CSC column).
    auto term_column = [this](term_t term) -> size_t {
        return static_cast<size_t>(std::ranges::lower_bound(term_ids_, term) -
                                   term_ids_.begin());
    };

    // Count entries per term, then prefix-sum into CSC offsets term_ptr_.
    term_ptr_.assign(n_terms + 1, 0);
    for (size_t j = 0; j < nnz; ++j) {
        term_ptr_[term_column(indices[j]) + 1]++;
    }
    for (size_t t = 0; t < n_terms; ++t) term_ptr_[t + 1] += term_ptr_[t];

    csc_cluster_.resize(nnz);
    csc_value_.resize(nnz * esz);
    std::vector<idx_t> cursor(term_ptr_.begin(), term_ptr_.end() - 1);
    for (size_t cluster = 0; cluster < n_clusters_; ++cluster) {
        const idx_t start = indptr[cluster];
        const idx_t end = indptr[cluster + 1];
        for (idx_t j = start; j < end; ++j) {
            const size_t col = term_column(indices[j]);
            const idx_t pos = cursor[col]++;
            csc_cluster_[pos] = static_cast<cluster_id_t>(cluster);
            std::copy_n(values + static_cast<size_t>(j) * esz, esz,
                        csc_value_.data() + static_cast<size_t>(pos) * esz);
        }
    }
}

template <class T>
void InvertedListClusters::score_summaries_typed(
    const term_t* q_idx, const T* q_val, size_t q_len,
    std::vector<float>& out) const {
    const T* csc_values = reinterpret_cast<const T*>(csc_value_.data());
    for (size_t i = 0; i < q_len; ++i) {
        const term_t term = q_idx[i];
        // Locate the query term among the summaries' distinct terms. Absent
        // terms (including any out of the summaries' range) contribute nothing.
        auto it = std::ranges::lower_bound(term_ids_, term);
        if (it == term_ids_.end() || *it != term) {
            continue;
        }
        const size_t col = static_cast<size_t>(it - term_ids_.begin());
        const float qv = static_cast<float>(q_val[i]);
        const idx_t start = term_ptr_[col];
        const idx_t end = term_ptr_[col + 1];
        for (idx_t j = start; j < end; ++j) {
            out[csc_cluster_[j]] += qv * static_cast<float>(csc_values[j]);
        }
    }
}

void InvertedListClusters::score_summaries_transposed(
    const term_t* q_idx, const uint8_t* q_val_bytes, size_t q_len,
    std::vector<float>& out) const {
    out.assign(n_clusters_, 0.0F);
    if (n_clusters_ == 0 || term_ids_.empty()) return;
    if (element_size_ == U32) {
        score_summaries_typed<float>(
            q_idx, reinterpret_cast<const float*>(q_val_bytes), q_len, out);
    } else if (element_size_ == U16) {
        score_summaries_typed<uint16_t>(
            q_idx, reinterpret_cast<const uint16_t*>(q_val_bytes), q_len, out);
    } else {
        score_summaries_typed<uint8_t>(q_idx, q_val_bytes, q_len, out);
    }
}

void InvertedListClusters::serialize(IOWriter* writer) const {
    size_t n_docs = docs_.size();
    writer->write(&n_docs, sizeof(size_t), 1);
    if (n_docs > 0) {
        writer->write(const_cast<idx_t*>(docs_.data()), sizeof(idx_t), n_docs);
    }
    size_t n_offsets = offsets_.size();
    writer->write(&n_offsets, sizeof(size_t), 1);
    if (n_offsets > 0) {
        writer->write(const_cast<idx_t*>(offsets_.data()), sizeof(idx_t),
                      n_offsets);
    }

    // Transposed (CSC) summary store.
    size_t n_clusters = n_clusters_;
    writer->write(&n_clusters, sizeof(size_t), 1);
    size_t element_size = element_size_;
    writer->write(&element_size, sizeof(size_t), 1);
    size_t n_terms = term_ids_.size();
    writer->write(&n_terms, sizeof(size_t), 1);
    if (n_terms > 0) {
        writer->write(const_cast<term_t*>(term_ids_.data()), sizeof(term_t),
                      n_terms);
        writer->write(const_cast<idx_t*>(term_ptr_.data()), sizeof(idx_t),
                      n_terms + 1);
    }
    size_t nnz = csc_cluster_.size();
    writer->write(&nnz, sizeof(size_t), 1);
    if (nnz > 0) {
        writer->write(const_cast<cluster_id_t*>(csc_cluster_.data()),
                      sizeof(cluster_id_t), nnz);
        writer->write(const_cast<uint8_t*>(csc_value_.data()), sizeof(uint8_t),
                      nnz * element_size_);
    }
}

void InvertedListClusters::deserialize(IOReader* reader) {
    size_t n_docs = 0;
    reader->read(&n_docs, sizeof(size_t), 1);
    if (n_docs > 0) {
        docs_.resize(n_docs);
        reader->read(docs_.data(), sizeof(idx_t), n_docs);
    }
    size_t n_offsets = 0;
    reader->read(&n_offsets, sizeof(size_t), 1);
    if (n_offsets > 0) {
        offsets_.resize(n_offsets);
        reader->read(offsets_.data(), sizeof(idx_t), n_offsets);
    }

    reader->read(&n_clusters_, sizeof(size_t), 1);
    reader->read(&element_size_, sizeof(size_t), 1);
    size_t n_terms = 0;
    reader->read(&n_terms, sizeof(size_t), 1);
    if (n_terms > 0) {
        term_ids_.resize(n_terms);
        reader->read(term_ids_.data(), sizeof(term_t), n_terms);
        term_ptr_.resize(n_terms + 1);
        reader->read(term_ptr_.data(), sizeof(idx_t), n_terms + 1);
    }
    size_t nnz = 0;
    reader->read(&nnz, sizeof(size_t), 1);
    if (nnz > 0) {
        csc_cluster_.resize(nnz);
        reader->read(csc_cluster_.data(), sizeof(cluster_id_t), nnz);
        csc_value_.resize(nnz * element_size_);
        reader->read(csc_value_.data(), sizeof(uint8_t), nnz * element_size_);
    }
}

}  // namespace nsparse
