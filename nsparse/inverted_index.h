/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef INVERTED_INDEX_H
#define INVERTED_INDEX_H

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "nsparse/invlists/inverted_lists.h"
#include "nsparse/io/io.h"
#include "nsparse/mmap_index.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/buf.h"

namespace nsparse {

// Two-phase search, after OpenSearch's neural_sparse_two_phase_processor with
// prune_type max_ratio. Query terms weighing less than prune_ratio times the
// query's largest weight sit out a first pass that collects the top
// k * expansion_rate documents (capped at max_window_size, never below k), and
// then rescore only those candidates. Lossy: a document only the low-weight
// terms match never becomes a candidate. prune_ratio 0 keeps every term in the
// first pass, which is the exact search.
struct InvertedSearchParameters : public SearchParameters {
    float prune_ratio = 0.4F;     // [0, 1]
    float expansion_rate = 5.0F;  // >= 1
    int max_window_size = 10000;  // > 0
    InvertedSearchParameters(float prune_ratio, float expansion_rate,
                             int max_window_size)
        : prune_ratio(prune_ratio),
          expansion_rate(expansion_rate),
          max_window_size(max_window_size) {}
    InvertedSearchParameters() = default;
};

class InvertedIndex : public MmapIndex, public IndexIO {
public:
    explicit InvertedIndex(int dim);

    InvertedIndex(const InvertedIndex&) = delete;
    InvertedIndex& operator=(const InvertedIndex&) = delete;
    void add(idx_t n, const offset_t* indptr, const term_t* indices,
             const float* values) override;
    void build() override;
    size_t num_vectors() const override { return num_vectors_; }

    // read_mcsr (the kMmap path) borrows vectors_ from the mapping without
    // going through add(), which is otherwise the only thing besides build()
    // that maintains num_vectors_. IDMapIndex::read_csr_and_ids cross-checks
    // this count against the id file's as soon as the delegate's read_csr
    // returns -- before build() runs -- so a mapped build reported 0 vectors
    // and every id map was rejected. Sync it here, as DiskSeismicIndexBase
    // does. (kInMemory routes through add(), which already set it, so
    // re-reading get_vectors() is then a harmless no-op.)
    void read_csr(const char* file_path,
                  Residency residency = Residency::kInMemory) override {
        MmapIndex::read_csr(file_path, residency);
        const auto* vectors = get_vectors();
        if (vectors != nullptr) {
            num_vectors_ = vectors->num_vectors();
        }
    }
    std::array<char, 4> id() const override { return name; }
    static constexpr std::array<char, 4> name = {'I', 'N', 'V', 'T'};
    // Bump whenever write_index's payload layout changes.
    static constexpr uint32_t kFormatVersion = 1;

    // Reads what write_index wrote, with the posting lists borrowing from a
    // mapping of `index_file` instead of being copied onto the heap. `pos` is
    // where the payload begins, past the header read_header consumed.
    static InvertedIndex* mmap_index(const IndexHeader& header,
                                     const char* index_file, size_t pos);

protected:
    auto search(idx_t n, const offset_t* indptr, const term_t* indices,
                const float* values, int k,
                SearchParameters* search_parameters = nullptr)
        -> pair_of_score_id_vectors_t override;

private:
    // IndexIO overrides
    [[nodiscard]] uint32_t format_version() const override {
        return kFormatVersion;
    }
    void write_index(IOWriter* io_writer) override;
    void read_index(IOReader* io_reader, const IndexHeader& header,
                    int io_flags) override;

    // `id_selector` may be null; when set, only member docs are returned.
    // A positive `prune_ratio` runs the two-phase search (see
    // InvertedSearchParameters), with `window` candidates from the first pass.
    auto single_query(const term_t* indices, const float* values, int size,
                      int k, const IDSelector* id_selector, float prune_ratio,
                      int window) -> pair_of_score_id_vector_t;
    std::unique_ptr<ArrayInvertedLists> inverted_lists_;
    // Tracked separately, since build() releases the vectors the base holds,
    // and serialised explicitly, since the posting lists hold no entry for a
    // document whose terms were all pruned.
    size_t num_vectors_ = 0;
    // Per-term max posting value, computed at build() time.
    // max_term_scores_[term_id] = max value in that term's posting list.
    // A Buf rather than a vector so a mapped index can borrow it in place.
    Buf<float> max_term_scores_;
};

}  // namespace nsparse
#endif  // INVERTED_INDEX_H
