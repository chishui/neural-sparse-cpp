/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/cluster/inverted_list_clusters.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "nsparse/io/buffered_io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/distance.h"
#include "nsparse/utils/mmap_cursor.h"

namespace {

nsparse::SparseVectors create_float_vectors(
    const std::vector<std::vector<nsparse::term_t>>& indices_list,
    const std::vector<std::vector<float>>& values_list, size_t dimension = 10) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = dimension});
    for (size_t i = 0; i < indices_list.size(); ++i) {
        const auto& indices = indices_list[i];
        const auto& values = values_list[i];
        vectors.add_vector(indices.data(), indices.size(),
                           reinterpret_cast<const uint8_t*>(values.data()),
                           values.size() * sizeof(float));
    }
    return vectors;
}

nsparse::SparseVectors create_uint8_vectors(
    const std::vector<std::vector<nsparse::term_t>>& indices_list,
    const std::vector<std::vector<uint8_t>>& values_list,
    size_t dimension = 10) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U8, .dimension = dimension});
    for (size_t i = 0; i < indices_list.size(); ++i) {
        const auto& indices = indices_list[i];
        const auto& values = values_list[i];
        vectors.add_vector(indices.data(), indices.size(), values.data(),
                           values.size());
    }
    return vectors;
}

nsparse::SparseVectors create_uint16_vectors(
    const std::vector<std::vector<nsparse::term_t>>& indices_list,
    const std::vector<std::vector<uint16_t>>& values_list,
    size_t dimension = 10) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U16, .dimension = dimension});
    for (size_t i = 0; i < indices_list.size(); ++i) {
        const auto& indices = indices_list[i];
        const auto& values = values_list[i];
        vectors.add_vector(indices.data(), indices.size(),
                           reinterpret_cast<const uint8_t*>(values.data()),
                           values.size() * sizeof(uint16_t));
    }
    return vectors;
}

// Recovers summary value[cluster][term] via the public scoring API: a one-hot
// query {term: 1} makes score_summaries_transposed return, per cluster, exactly
// that cluster's summary value at `term` (0 if absent). Replaces the old
// summaries().get_dense_vector*() inspection now that only the transpose is
// stored. `q_val_t` is the query value type matching the summaries' element
// width (float / uint16_t / uint8_t).
template <class q_val_t>
std::vector<float> summary_values_at_term(
    const nsparse::InvertedListClusters& clusters, nsparse::term_t term) {
    std::vector<nsparse::term_t> q_idx = {term};
    std::vector<q_val_t> q_val = {static_cast<q_val_t>(1)};
    std::vector<float> out;
    clusters.score_summaries_transposed(
        q_idx.data(), reinterpret_cast<const uint8_t*>(q_val.data()),
        q_idx.size(), out);
    return out;
}

}  // namespace

// Constructor tests
TEST(InvertedListClusters, default_constructor) {
    nsparse::InvertedListClusters clusters;
    ASSERT_EQ(clusters.cluster_size(), 0);
}

TEST(InvertedListClusters, constructor_with_empty_docs) {
    std::vector<std::vector<nsparse::idx_t>> docs = {};
    nsparse::InvertedListClusters clusters(docs);
    ASSERT_EQ(clusters.cluster_size(), 0);
}

TEST(InvertedListClusters, constructor_with_single_cluster) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1, 2}};
    nsparse::InvertedListClusters clusters(docs);

    auto doc_span = clusters.get_docs(0);
    ASSERT_EQ(doc_span.size(), 3);
    ASSERT_EQ(doc_span[0], 0);
    ASSERT_EQ(doc_span[1], 1);
    ASSERT_EQ(doc_span[2], 2);
}

TEST(InvertedListClusters, constructor_with_multiple_clusters) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2, 3, 4}, {5}};
    nsparse::InvertedListClusters clusters(docs);

    auto doc_span0 = clusters.get_docs(0);
    ASSERT_EQ(doc_span0.size(), 2);
    ASSERT_EQ(doc_span0[0], 0);
    ASSERT_EQ(doc_span0[1], 1);

    auto doc_span1 = clusters.get_docs(1);
    ASSERT_EQ(doc_span1.size(), 3);
    ASSERT_EQ(doc_span1[0], 2);
    ASSERT_EQ(doc_span1[1], 3);
    ASSERT_EQ(doc_span1[2], 4);

    auto doc_span2 = clusters.get_docs(2);
    ASSERT_EQ(doc_span2.size(), 1);
    ASSERT_EQ(doc_span2[0], 5);
}

TEST(InvertedListClusters, constructor_with_empty_cluster) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {}, {2}};
    nsparse::InvertedListClusters clusters(docs);

    auto doc_span0 = clusters.get_docs(0);
    ASSERT_EQ(doc_span0.size(), 2);

    auto doc_span1 = clusters.get_docs(1);
    ASSERT_EQ(doc_span1.size(), 0);

    auto doc_span2 = clusters.get_docs(2);
    ASSERT_EQ(doc_span2.size(), 1);
}

// Move tests. The type is move-only: its arrays are Buf, which may borrow memory
// it does not own, so copying is deleted rather than silently deep-copying.
static_assert(!std::is_copy_constructible_v<nsparse::InvertedListClusters>);
static_assert(!std::is_copy_assignable_v<nsparse::InvertedListClusters>);

TEST(InvertedListClusters, move_constructor_without_summaries) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2, 3}};
    nsparse::InvertedListClusters original(docs);

    nsparse::InvertedListClusters moved(std::move(original));

    auto doc_span = moved.get_docs(0);
    ASSERT_EQ(doc_span.size(), 2);
    ASSERT_EQ(doc_span[0], 0);
    ASSERT_EQ(doc_span[1], 1);
}

TEST(InvertedListClusters, move_constructor_with_summaries) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2}};
    nsparse::InvertedListClusters original(docs);

    auto vectors = create_float_vectors(
        {{0, 1}, {0, 2}, {1, 2}}, {{1.0F, 2.0F}, {1.5F, 1.0F}, {3.0F, 2.0F}});
    original.summarize(&vectors, 1.0F);

    nsparse::InvertedListClusters moved(std::move(original));

    ASSERT_EQ(moved.cluster_size(), 2);
    // The transpose must survive the move intact: cluster 0 has term 0, cluster
    // 1 has term 2.
    auto term0 = summary_values_at_term<float>(moved, 0);
    ASSERT_EQ(term0.size(), 2U);
    ASSERT_GT(term0[0], 0.0F);
}

TEST(InvertedListClusters, move_assignment_without_summaries) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2, 3}};
    nsparse::InvertedListClusters original(docs);

    nsparse::InvertedListClusters moved;
    moved = std::move(original);

    auto doc_span = moved.get_docs(1);
    ASSERT_EQ(doc_span.size(), 2);
    ASSERT_EQ(doc_span[0], 2);
    ASSERT_EQ(doc_span[1], 3);
}

// Move-assigning over a populated instance must release what it held rather than
// leak it, and take on the source's contents.
TEST(InvertedListClusters, move_assignment_over_populated) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2, 3}};
    nsparse::InvertedListClusters original(docs);

    std::vector<std::vector<nsparse::idx_t>> other_docs = {{7, 8, 9}};
    nsparse::InvertedListClusters target(other_docs);
    target = std::move(original);

    auto doc_span = target.get_docs(0);
    ASSERT_EQ(doc_span.size(), 2);
    ASSERT_EQ(doc_span[0], 0);
    ASSERT_EQ(doc_span[1], 1);
}

// Summarize tests
TEST(InvertedListClusters, summarize_float_single_cluster) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}};
    nsparse::InvertedListClusters clusters(docs);

    // Vector 0: term 0 -> 1.0, term 1 -> 2.0
    // Vector 1: term 0 -> 3.0, term 1 -> 1.0
    // Summary should have max: term 0 -> 3.0, term 1 -> 2.0
    auto vectors =
        create_float_vectors({{0, 1}, {0, 1}}, {{1.0F, 2.0F}, {3.0F, 1.0F}});

    clusters.summarize(&vectors, 1.0F);

    ASSERT_EQ(clusters.cluster_size(), 1);
    ASSERT_FLOAT_EQ(summary_values_at_term<float>(clusters, 0)[0],
                    3.0F);  // max(1.0, 3.0)
    ASSERT_FLOAT_EQ(summary_values_at_term<float>(clusters, 1)[0],
                    2.0F);  // max(2.0, 1.0)
}

TEST(InvertedListClusters, summarize_float_multiple_clusters) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2}};
    nsparse::InvertedListClusters clusters(docs);

    // Cluster 0: vectors 0,1 -> term 0: max(1.0, 2.0)=2.0
    // Cluster 1: vector 2 -> term 1: 3.0
    auto vectors =
        create_float_vectors({{0}, {0}, {1}}, {{1.0F}, {2.0F}, {3.0F}});

    clusters.summarize(&vectors, 1.0F);

    ASSERT_EQ(clusters.cluster_size(), 2);

    // Cluster 0 has term 0 -> 2.0; cluster 1 has term 1 -> 3.0.
    ASSERT_FLOAT_EQ(summary_values_at_term<float>(clusters, 0)[0],
                    2.0F);  // max(1.0, 2.0)
    ASSERT_FLOAT_EQ(summary_values_at_term<float>(clusters, 1)[1], 3.0F);
}

TEST(InvertedListClusters, summarize_uint8) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}};
    nsparse::InvertedListClusters clusters(docs);

    // Vector 0: term 0 -> 10, term 1 -> 20
    // Vector 1: term 0 -> 30, term 1 -> 10
    // Summary: term 0 -> max(10,30)=30, term 1 -> max(20,10)=20
    auto vectors = create_uint8_vectors({{0, 1}, {0, 1}}, {{10, 20}, {30, 10}});

    clusters.summarize(&vectors, 1.0F);

    ASSERT_EQ(clusters.cluster_size(), 1);

    ASSERT_FLOAT_EQ(summary_values_at_term<uint8_t>(clusters, 0)[0],
                    30.0F);  // max(10, 30)
    ASSERT_FLOAT_EQ(summary_values_at_term<uint8_t>(clusters, 1)[0],
                    20.0F);  // max(20, 10)
}

TEST(InvertedListClusters, summarize_uint16) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}};
    nsparse::InvertedListClusters clusters(docs);

    // Vector 0: term 0 -> 100, term 1 -> 200
    // Vector 1: term 0 -> 300, term 1 -> 100
    // Summary: term 0 -> max(100,300)=300, term 1 -> max(200,100)=200
    auto vectors =
        create_uint16_vectors({{0, 1}, {0, 1}}, {{100, 200}, {300, 100}});

    clusters.summarize(&vectors, 1.0F);

    ASSERT_EQ(clusters.cluster_size(), 1);

    ASSERT_FLOAT_EQ(summary_values_at_term<uint16_t>(clusters, 0)[0],
                    300.0F);  // max(100, 300)
    ASSERT_FLOAT_EQ(summary_values_at_term<uint16_t>(clusters, 1)[0],
                    200.0F);  // max(200, 100)
}

TEST(InvertedListClusters, summarize_with_alpha_pruning) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0}};
    nsparse::InvertedListClusters clusters(docs);

    // Vector with terms: 0->10.0, 1->5.0, 2->3.0, 3->2.0
    // Total sum = 20.0
    // With alpha=0.5, should keep terms until cumsum >= 10.0
    // Sorted by value desc: term0(10), term1(5), term2(3), term3(2)
    // cumsum after term0: 10/20 = 0.5 >= 0.5, so keep only term0
    auto vectors =
        create_float_vectors({{0, 1, 2, 3}}, {{10.0F, 5.0F, 3.0F, 2.0F}});

    clusters.summarize(&vectors, 0.5F);

    ASSERT_EQ(clusters.cluster_size(), 1);
    // Only term 0 should survive alpha pruning (non-zero); the pruned terms
    // contribute nothing.
    ASSERT_GT(summary_values_at_term<float>(clusters, 0)[0], 0.0F);
    ASSERT_FLOAT_EQ(summary_values_at_term<float>(clusters, 1)[0], 0.0F);
}

TEST(InvertedListClusters, summarize_replaces_existing) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0}};
    nsparse::InvertedListClusters clusters(docs);

    auto vectors1 = create_float_vectors({{0}}, {{1.0F}});
    clusters.summarize(&vectors1, 1.0F);
    ASSERT_EQ(clusters.cluster_size(), 1);

    auto vectors2 = create_float_vectors({{0}, {1}}, {{2.0F}, {3.0F}});
    std::vector<std::vector<nsparse::idx_t>> docs2 = {{0, 1}};
    nsparse::InvertedListClusters clusters2(docs2);
    clusters2.summarize(&vectors2, 1.0F);
    ASSERT_EQ(clusters2.cluster_size(), 1);
}

// Serialization tests
TEST(InvertedListClusters, serialize_deserialize_empty) {
    nsparse::InvertedListClusters original;

    nsparse::BufferedIOWriter writer;
    original.serialize(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::InvertedListClusters loaded;
    loaded.deserialize(&reader);

    ASSERT_EQ(loaded.cluster_size(), 0);
}

TEST(InvertedListClusters, serialize_deserialize_without_summaries) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1, 2}, {3, 4}};
    nsparse::InvertedListClusters original(docs);

    nsparse::BufferedIOWriter writer;
    original.serialize(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::InvertedListClusters loaded;
    loaded.deserialize(&reader);

    ASSERT_EQ(loaded.cluster_size(), 0);  // No summaries

    auto doc_span0 = loaded.get_docs(0);
    ASSERT_EQ(doc_span0.size(), 3);
    ASSERT_EQ(doc_span0[0], 0);
    ASSERT_EQ(doc_span0[1], 1);
    ASSERT_EQ(doc_span0[2], 2);

    auto doc_span1 = loaded.get_docs(1);
    ASSERT_EQ(doc_span1.size(), 2);
    ASSERT_EQ(doc_span1[0], 3);
    ASSERT_EQ(doc_span1[1], 4);
}

TEST(InvertedListClusters, serialize_deserialize_with_summaries) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2}};
    nsparse::InvertedListClusters original(docs);

    auto vectors = create_float_vectors(
        {{0, 1}, {0, 2}, {1, 2}}, {{1.0F, 2.0F}, {1.5F, 1.0F}, {3.0F, 2.0F}});
    original.summarize(&vectors, 1.0F);

    nsparse::BufferedIOWriter writer;
    original.serialize(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::InvertedListClusters loaded;
    loaded.deserialize(&reader);

    ASSERT_EQ(loaded.cluster_size(), 2);

    auto doc_span0 = loaded.get_docs(0);
    ASSERT_EQ(doc_span0.size(), 2);

    auto doc_span1 = loaded.get_docs(1);
    ASSERT_EQ(doc_span1.size(), 1);
}

// Move constructor/assignment tests
TEST(InvertedListClusters, move_constructor) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2, 3}};
    nsparse::InvertedListClusters original(docs);

    nsparse::InvertedListClusters moved(std::move(original));

    auto doc_span = moved.get_docs(0);
    ASSERT_EQ(doc_span.size(), 2);
    ASSERT_EQ(doc_span[0], 0);
    ASSERT_EQ(doc_span[1], 1);
}

TEST(InvertedListClusters, move_assignment) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2, 3}};
    nsparse::InvertedListClusters original(docs);

    nsparse::InvertedListClusters moved;
    moved = std::move(original);

    auto doc_span = moved.get_docs(1);
    ASSERT_EQ(doc_span.size(), 2);
    ASSERT_EQ(doc_span[0], 2);
    ASSERT_EQ(doc_span[1], 3);
}

// Verifies that the query-driven transposed summary scorer
// (score_summaries_transposed, CSC) produces exactly the same per-cluster
// scores as the reference dense-gather over the CSR summaries. This is the
// data-consistency check between the CSR and CSC representations.
TEST(InvertedListClusters, transposed_scoring_matches_reference) {
    // Two clusters with overlapping and distinct summary terms.
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2}};
    nsparse::InvertedListClusters clusters(docs);
    auto vectors = create_float_vectors(
        {{0, 3}, {1, 3}, {2, 5}}, {{1.0F, 2.0F}, {4.0F, 1.0F}, {3.0F, 2.0F}},
        /*dimension=*/10);
    clusters.summarize(&vectors, 1.0F);

    const size_t n_clusters = clusters.cluster_size();
    ASSERT_EQ(n_clusters, 2U);

    // Reference summary matrix: recover value[cluster][term] via the one-hot
    // helper, so we can independently compute the expected query dot product
    // per cluster without inspecting the internal storage.
    const size_t dim = 10;
    std::vector<std::vector<float>> summary(n_clusters,
                                            std::vector<float>(dim, 0.0F));
    for (nsparse::term_t t = 0; t < dim; ++t) {
        auto vals = summary_values_at_term<float>(clusters, t);
        for (size_t c = 0; c < n_clusters; ++c) summary[c][t] = vals[c];
    }
    auto reference_scores = [&](const std::vector<nsparse::term_t>& q_idx,
                                const std::vector<float>& q_val) {
        std::vector<float> out(n_clusters, 0.0F);
        for (size_t c = 0; c < n_clusters; ++c) {
            for (size_t i = 0; i < q_idx.size(); ++i) {
                if (q_idx[i] < dim) out[c] += q_val[i] * summary[c][q_idx[i]];
            }
        }
        return out;
    };

    // Query terms exercising: a shared term (3), a term unique to one summary
    // (0), a term absent from all summaries (7), and an out-of-range term id
    // (9999 > dim) that must be safely ignored.
    std::vector<std::vector<nsparse::term_t>> queries = {
        {3}, {0, 5}, {0, 3, 5}, {7}, {9999}, {3, 7, 9999}};
    std::vector<std::vector<float>> query_values = {
        {2.0F}, {1.5F, 0.5F}, {1.0F, 1.0F, 1.0F},
        {3.0F}, {2.0F},       {2.0F, 1.0F, 1.0F}};

    for (size_t q = 0; q < queries.size(); ++q) {
        std::vector<float> expected =
            reference_scores(queries[q], query_values[q]);

        std::vector<float> got;
        clusters.score_summaries_transposed(
            queries[q].data(),
            reinterpret_cast<const uint8_t*>(query_values[q].data()),
            queries[q].size(), got);

        ASSERT_EQ(got.size(), expected.size());
        for (size_t c = 0; c < got.size(); ++c) {
            ASSERT_FLOAT_EQ(got[c], expected[c])
                << "query " << q << ", cluster " << c;
        }
    }
}

// ---------------------------------------------------------------------------
// The serialized layout, walked by both readers.
//
// What is covered is that the two agree, not that either is defensive: neither
// checks what it reads past the bounds of the mapping, so a self-inconsistent
// file is read out of bounds at search time. Deliberate -- scanning the summaries
// to rule it out costs more than the mapped load itself.
// ---------------------------------------------------------------------------

namespace {

// Lays out a payload field by field, so a test can plant a value serialize()
// would never produce. Padding runs off the position within the payload, where
// both readers start.
class ClusterPayload {
public:
    ClusterPayload& scalar(size_t value) {
        return append(&value, sizeof(value));
    }

    template <class T>
    ClusterPayload& array(const std::vector<T>& values) {
        return pad(alignof(T)).append(values.data(), values.size() * sizeof(T));
    }

    // Reinterpreted `width` at a time on read, so padded to that, not to 1.
    ClusterPayload& values(const std::vector<uint8_t>& bytes, size_t width) {
        return pad(width).append(bytes.data(), bytes.size());
    }

    [[nodiscard]] const std::vector<uint8_t>& data() const { return bytes_; }

private:
    // A zero alignment inserts nothing, as a file claiming a zero element width
    // would. The reader still has to reject the width, not modulo by it.
    ClusterPayload& pad(size_t alignment) {
        while (alignment != 0 && bytes_.size() % alignment != 0) {
            bytes_.push_back(0);
        }
        return *this;
    }

    ClusterPayload& append(const void* src, size_t count) {
        const auto* first = static_cast<const uint8_t*>(src);
        bytes_.insert(bytes_.end(), first, first + count);
        return *this;
    }

    std::vector<uint8_t> bytes_;
};

// Mirrors InvertedListClusters::cluster_id_t, which is private.
using stored_cluster_id_t = uint16_t;

// One cluster over two docs, one summary term, one entry. Parameterized so a test
// can make exactly one field inconsistent.
ClusterPayload well_formed_payload(
    size_t n_clusters = 1, size_t element_size = nsparse::U32,
    const std::vector<nsparse::idx_t>& offsets = {0, 2},
    const std::vector<nsparse::term_t>& term_ids = {5},
    const std::vector<nsparse::idx_t>& term_ptr = {0, 1},
    const std::vector<stored_cluster_id_t>& csc_cluster = {0}) {
    ClusterPayload payload;
    payload.scalar(2).array<nsparse::idx_t>({0, 1});
    payload.scalar(offsets.size()).array(offsets);
    payload.scalar(n_clusters);
    payload.scalar(element_size);
    payload.scalar(term_ids.size());
    if (!term_ids.empty()) {
        payload.array(term_ids).array(term_ptr);
    }
    payload.scalar(csc_cluster.size()).array(csc_cluster);
    payload.values(std::vector<uint8_t>(csc_cluster.size() * element_size, 1),
                   element_size);
    return payload;
}

// Both paths walk one layout, so a check missing from either is a hole.
void expect_both_readers_reject(const ClusterPayload& payload) {
    nsparse::BufferedIOReader reader(payload.data());
    nsparse::InvertedListClusters copied;
    EXPECT_THROW(copied.deserialize(&reader), std::runtime_error);

    nsparse::MmapCursor cursor(payload.data().data(), payload.data().size());
    nsparse::InvertedListClusters mapped;
    EXPECT_THROW(mapped.mmap_deserialize(&cursor), std::runtime_error);
}

}  // namespace

// Both readers consume a hand-built payload and land on the same byte.
TEST(InvertedListClustersLayout, accepts_a_well_formed_payload) {
    const auto payload = well_formed_payload();

    nsparse::BufferedIOReader reader(payload.data());
    nsparse::InvertedListClusters copied;
    ASSERT_NO_THROW(copied.deserialize(&reader));
    ASSERT_EQ(copied.cluster_size(), 1);
    ASSERT_EQ(reader.pos(), payload.data().size());

    nsparse::MmapCursor cursor(payload.data().data(), payload.data().size());
    nsparse::InvertedListClusters mapped;
    ASSERT_NO_THROW(mapped.mmap_deserialize(&cursor));
    ASSERT_EQ(mapped.cluster_size(), 1);
    ASSERT_EQ(cursor.pos(), payload.data().size());
}

// Reaches padding_for() as an alignment, which used to divide by zero on the
// mapped path while the copying path skipped the padding entirely.
TEST(InvertedListClustersLayout, rejects_a_zero_element_width) {
    expect_both_readers_reject(well_formed_payload(
        /*n_clusters=*/1, /*element_size=*/0, {0, 2}, {5}, {0, 1}, {0}));
}
