/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/seismic_scalar_quantized_index.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/id_selector.h"
#include "nsparse/index.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/index_io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse {
namespace {

// Testable subclass that exposes protected members
class TestableSeismicSQIndex : public SeismicScalarQuantizedIndex {
public:
    using SeismicScalarQuantizedIndex::add;
    using SeismicScalarQuantizedIndex::SeismicScalarQuantizedIndex;

    TestableSeismicSQIndex(QuantizerType qt, float vmin, float vmax, int lambda,
                           int beta, float alpha, int dim)
        : SeismicScalarQuantizedIndex(
              qt, vmin, vmax, {.lambda = lambda, .beta = beta, .alpha = alpha},
              dim) {}

    std::vector<InvertedListClusters>& get_clustered_inverted_lists() {
        return clustered_inverted_lists;
    }

    // Helper to add docs from map format: {{term: value, ...}, ...}
    void add_docs(const std::vector<std::map<int, float>>& docs) {
        std::vector<idx_t> indptr;
        std::vector<term_t> indices;
        std::vector<float> values;

        indptr.push_back(0);
        for (const auto& doc : docs) {
            for (const auto& [term, value] : doc) {
                indices.push_back(static_cast<term_t>(term));
                values.push_back(value);
            }
            indptr.push_back(static_cast<idx_t>(indices.size()));
        }

        SeismicScalarQuantizedIndex::add(static_cast<idx_t>(docs.size()),
                                         indptr.data(), indices.data(),
                                         values.data());
    }
};

// ============== build() tests ==============

TEST(SeismicSQIndexBuild, build_creates_clustered_inverted_lists_8bit) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 2, 2, 0.5F,
                                 3);

    index.add_docs({{{0, 1.0F}, {1, 0.5F}},
                    {{0, 0.8F}, {1, 0.6F}, {2, 0.7F}},
                    {{1, 0.9F}, {2, 0.4F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    EXPECT_EQ(inv_lists.size(), 3);
    EXPECT_EQ(inv_lists.at(0).cluster_size(), 2);
    EXPECT_EQ(inv_lists.at(1).cluster_size(), 2);
    EXPECT_EQ(inv_lists.at(2).cluster_size(), 2);
}

TEST(SeismicSQIndexBuild, build_creates_clustered_inverted_lists_16bit) {
    TestableSeismicSQIndex index(QuantizerType::QT_16bit, 0.0F, 1.0F, 2, 2,
                                 0.5F, 3);

    index.add_docs({{{0, 1.0F}, {1, 0.5F}},
                    {{0, 0.8F}, {1, 0.6F}, {2, 0.7F}},
                    {{1, 0.9F}, {2, 0.4F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    EXPECT_EQ(inv_lists.size(), 3);
    EXPECT_EQ(inv_lists.at(0).cluster_size(), 2);
    EXPECT_EQ(inv_lists.at(1).cluster_size(), 2);
    EXPECT_EQ(inv_lists.at(2).cluster_size(), 2);
}

TEST(SeismicSQIndexBuild, build_with_single_vector) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 1, 1, 0.5F,
                                 3);

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    EXPECT_EQ(inv_lists.size(), 3);
    EXPECT_EQ(inv_lists[0].cluster_size(), 1);
    EXPECT_EQ(inv_lists[1].cluster_size(), 1);
    EXPECT_EQ(inv_lists[2].cluster_size(), 0);
}

TEST(SeismicSQIndexBuild, build_populates_inverted_lists_for_each_term) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 4);

    index.add_docs({{{0, 1.0F}}, {{1, 1.0F}}, {{2, 1.0F}}, {{3, 1.0F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    EXPECT_EQ(inv_lists.size(), 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(inv_lists[i].cluster_size(), 1);
    }
}

TEST(SeismicSQIndexBuild, build_with_multiple_docs_same_term) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 3);

    index.add_docs({{{0, 1.0F}}, {{0, 0.9F}}, {{0, 0.8F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    EXPECT_EQ(inv_lists.size(), 3);
    EXPECT_EQ(inv_lists[0].cluster_size(), 2);
    EXPECT_EQ(inv_lists[1].cluster_size(), 0);
    EXPECT_EQ(inv_lists[2].cluster_size(), 0);
}

// ============== Constructor tests ==============

TEST(SeismicSQIndexConstructor, constructor_with_dim_only) {
    SeismicScalarQuantizedIndex index(100);
    EXPECT_EQ(index.get_dimension(), 100);
}

TEST(SeismicSQIndexConstructor, constructor_with_all_params_8bit) {
    SeismicScalarQuantizedIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                      {.lambda = 5, .beta = 3, .alpha = 0.6F},
                                      50);
    EXPECT_EQ(index.get_dimension(), 50);
    EXPECT_EQ(index.get_scalar_quantizer().get_quantizer_type(),
              QuantizerType::QT_8bit);
}

TEST(SeismicSQIndexConstructor, constructor_with_all_params_16bit) {
    SeismicScalarQuantizedIndex index(QuantizerType::QT_16bit, 0.0F, 2.0F,
                                      {.lambda = 5, .beta = 3, .alpha = 0.6F},
                                      50);
    EXPECT_EQ(index.get_dimension(), 50);
    EXPECT_EQ(index.get_scalar_quantizer().get_quantizer_type(),
              QuantizerType::QT_16bit);
}

// ============== add() tests ==============

TEST(SeismicSQIndexAdd, add_creates_vectors) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 5, 2, 0.5F,
                                 5);

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});

    EXPECT_NE(index.get_vectors(), nullptr);
    EXPECT_EQ(index.get_vectors()->num_vectors(), 1);
}

TEST(SeismicSQIndexAdd, add_multiple_vectors) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 5, 2, 0.5F,
                                 5);

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}, {{2, 0.8F}, {3, 0.6F}}});

    EXPECT_EQ(index.get_vectors()->num_vectors(), 2);
}

TEST(SeismicSQIndexAdd, add_quantizes_values_8bit) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 5, 2, 0.5F,
                                 5);

    index.add_docs({{{0, 1.0F}, {2, 0.5F}}});

    const auto* vecs = index.get_vectors();
    EXPECT_EQ(vecs->get_element_size(), U8);
}

TEST(SeismicSQIndexAdd, add_quantizes_values_16bit) {
    TestableSeismicSQIndex index(QuantizerType::QT_16bit, 0.0F, 1.0F, 5, 2,
                                 0.5F, 5);

    index.add_docs({{{0, 1.0F}, {2, 0.5F}}});

    const auto* vecs = index.get_vectors();
    EXPECT_EQ(vecs->get_element_size(), U16);
}

// ============== get_vectors() tests ==============

TEST(SeismicSQIndexGetVectors, returns_null_before_add) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 5, 2, 0.5F,
                                 5);
    EXPECT_EQ(index.get_vectors(), nullptr);
}

TEST(SeismicSQIndexGetVectors, returns_vectors_after_add) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 5, 2, 0.5F,
                                 5);

    index.add_docs({{{0, 1.0F}}});

    EXPECT_NE(index.get_vectors(), nullptr);
}

// ============== SeismicSQSearchParameters tests ==============

TEST(SeismicSQSearchParameters, constructor_sets_values) {
    SeismicSQSearchParameters params(0.0F, 1.0F, 20, 0.5F);
    EXPECT_FLOAT_EQ(params.vmin, 0.0F);
    EXPECT_FLOAT_EQ(params.vmax, 1.0F);
    EXPECT_EQ(params.cut, 20);
    EXPECT_FLOAT_EQ(params.heap_factor, 0.5F);
}

// ============== search() tests ==============

TEST(SeismicSQIndexSearch, search_returns_empty_when_no_vectors) {
    SeismicScalarQuantizedIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                      {.lambda = 5, .beta = 2, .alpha = 0.5F},
                                      5);
    Index* idx = &index;

    std::vector<idx_t> query_indptr = {0, 2};
    std::vector<term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 0.5F};
    std::vector<idx_t> labels(5, -1);
    std::vector<float> distances(5, -1.0F);

    SeismicSearchParameters params(10, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 5, distances.data(), labels.data(),
                &params);

    for (const auto& label : labels) {
        EXPECT_EQ(label, -1);
    }
}

TEST(SeismicSQIndexSearch, search_finds_matching_doc_8bit) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 3);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    SeismicSearchParameters params(5, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 0);
}

TEST(SeismicSQIndexSearch, search_finds_matching_doc_16bit) {
    TestableSeismicSQIndex index(QuantizerType::QT_16bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 3);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    SeismicSearchParameters params(5, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 0);
}

TEST(SeismicSQIndexSearch, search_multiple_queries) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 4);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}, {{2, 0.8F}, {3, 0.6F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1, 2};
    std::vector<term_t> query_indices = {0, 2};
    std::vector<float> query_values = {1.0F, 1.0F};
    std::vector<idx_t> labels(2, -1);
    std::vector<float> distances(2, -1.0F);

    SeismicSearchParameters params(5, 1.0F);
    idx->search(2, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
}

TEST(SeismicSQIndexSearch, search_respects_k_limit) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 3);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}}, {{0, 0.9F}}, {{0, 0.8F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(2, -1);
    std::vector<float> distances(2, -1.0F);

    SeismicSearchParameters params(5, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 2, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
}

TEST(SeismicSQIndexSearch, search_with_default_parameters) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 3);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 2};
    std::vector<term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 0.5F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    SeismicSearchParameters params(10, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 0);
}

// ============== Complex search tests ==============

TEST(SeismicSQIndexSearch, lambda_prunes_posting_list) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 2, 1, 0.5F,
                                 3);
    Index* idx = &index;

    index.add_docs({{{0, 0.1F}}, {{0, 0.2F}}, {{0, 0.3F}}, {{0, 0.4F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    size_t total_docs = 0;
    for (size_t c = 0; c < inv_lists[0].cluster_size(); ++c) {
        total_docs += inv_lists[0].get_docs(c).size();
    }
    EXPECT_EQ(total_docs, 2);

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(4, -1);
    std::vector<float> distances(4, -1.0F);

    SeismicSearchParameters params(5, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 4, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 3);
    EXPECT_EQ(labels[1], 2);
}

TEST(SeismicSQIndexSearch, cut_prunes_query_tokens) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 4);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}}, {{1, 1.0F}}, {{2, 1.0F}}, {{3, 1.0F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 3};
    std::vector<term_t> query_indices = {0, 1, 2};
    std::vector<float> query_values = {0.1F, 0.5F, 0.9F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    SeismicSearchParameters params(1, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 2);
}

TEST(SeismicSQIndexSearch, large_heap_factor_includes_all_clusters) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 5);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}},
                    {{0, 0.9F}},
                    {{1, 0.8F}, {2, 0.3F}},
                    {{2, 0.7F}},
                    {{3, 0.6F}},
                    {{0, 0.1F}, {3, 0.4F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 3};
    std::vector<term_t> query_indices = {0, 1, 2};
    std::vector<float> query_values = {1.0F, 0.8F, 0.5F};
    std::vector<idx_t> labels(6, -1);
    std::vector<float> distances(6, -1.0F);

    SeismicSearchParameters params(5, 1000.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 6, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
    EXPECT_EQ(labels[2], 2);
    EXPECT_EQ(labels[3], 3);
    EXPECT_EQ(labels[4], 5);
}

// ============== Search with SeismicSQSearchParameters tests ==============

TEST(SeismicSQIndexSearch, search_with_sq_search_parameters) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 3);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    SeismicSQSearchParameters params(0.0F, 1.0F, 5, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 0);
}

// ============== write_index/read_index tests ==============

TEST(SeismicSQIndexIO, write_and_read_empty_index) {
    SeismicScalarQuantizedIndex original(
        QuantizerType::QT_8bit, 0.0F, 1.0F,
        {.lambda = 5, .beta = 2, .alpha = 0.5F}, 100);

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_dimension(), 100);
    EXPECT_EQ(loaded->id(), original.id());
    EXPECT_EQ(loaded->get_vectors(), nullptr);

    delete loaded;
}

TEST(SeismicSQIndexIO, write_and_read_with_vectors) {
    TestableSeismicSQIndex original(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                    0.5F, 5);

    original.add_docs({{{0, 1.0F}, {1, 0.5F}}, {{2, 0.8F}, {3, 0.6F}}});

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_dimension(), 5);

    const auto* vecs = loaded->get_vectors();
    ASSERT_NE(vecs, nullptr);
    EXPECT_EQ(vecs->num_vectors(), 2);

    delete loaded;
}

TEST(SeismicSQIndexIO, write_and_read_built_index) {
    TestableSeismicSQIndex original(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                    0.5F, 4);

    original.add_docs({{{0, 1.0F}, {1, 0.5F}}, {{2, 0.8F}, {3, 0.6F}}});
    original.build();

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_dimension(), 4);
    EXPECT_EQ(loaded->get_vectors()->num_vectors(), 2);

    delete loaded;
}

TEST(SeismicSQIndexIO, write_and_read_search_produces_same_results) {
    TestableSeismicSQIndex original(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                    0.5F, 4);

    original.add_docs({{{0, 1.0F}, {1, 0.5F}},
                       {{0, 0.8F}, {2, 0.6F}},
                       {{1, 0.9F}, {3, 0.7F}}});
    original.build();

    std::vector<idx_t> query_indptr = {0, 2};
    std::vector<term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 0.8F};
    std::vector<idx_t> labels_original(3, -1);
    std::vector<float> distances_original(3, -1.0F);

    SeismicSearchParameters params(5, 1000.0F);
    Index* idx_original = &original;
    idx_original->search(1, query_indptr.data(), query_indices.data(),
                         query_values.data(), 3, distances_original.data(),
                         labels_original.data(), &params);

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    std::vector<idx_t> labels_loaded(3, -1);
    std::vector<float> distances_loaded(3, -1.0F);
    loaded->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 3, distances_loaded.data(),
                   labels_loaded.data(), &params);

    EXPECT_EQ(labels_original[0], labels_loaded[0]);
    EXPECT_EQ(labels_original[1], labels_loaded[1]);
    EXPECT_EQ(labels_original[2], labels_loaded[2]);

    delete loaded;
}

TEST(SeismicSQIndexIO, write_and_read_preserves_quantizer_type_8bit) {
    TestableSeismicSQIndex original(QuantizerType::QT_8bit, 0.1F, 0.9F, 10, 2,
                                    0.5F, 5);

    original.add_docs({{{0, 0.5F}}});

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    auto* loaded =
        dynamic_cast<SeismicScalarQuantizedIndex*>(read_index(&reader));

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_scalar_quantizer().get_quantizer_type(),
              QuantizerType::QT_8bit);
    EXPECT_FLOAT_EQ(loaded->get_scalar_quantizer().get_min(), 0.1F);
    EXPECT_FLOAT_EQ(loaded->get_scalar_quantizer().get_max(), 0.9F);

    delete loaded;
}

TEST(SeismicSQIndexIO, write_and_read_preserves_quantizer_type_16bit) {
    TestableSeismicSQIndex original(QuantizerType::QT_16bit, 0.0F, 2.0F, 10, 2,
                                    0.5F, 5);

    original.add_docs({{{0, 1.0F}}});

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    auto* loaded =
        dynamic_cast<SeismicScalarQuantizedIndex*>(read_index(&reader));

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_scalar_quantizer().get_quantizer_type(),
              QuantizerType::QT_16bit);
    EXPECT_FLOAT_EQ(loaded->get_scalar_quantizer().get_min(), 0.0F);
    EXPECT_FLOAT_EQ(loaded->get_scalar_quantizer().get_max(), 2.0F);

    delete loaded;
}

// ============== Integration-style tests ==============

TEST(SeismicSQIndexSearch, heap_factor_controls_result_count_large_dataset) {
    constexpr int kDocCount = 100;
    constexpr int kDimension = 5000;
    constexpr int kLambda = 100;
    constexpr int kBeta = 10;
    constexpr float kAlpha = 0.5F;

    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, kLambda,
                                 kBeta, kAlpha, kDimension);

    std::vector<std::map<int, float>> docs;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> value_dist(0.1F, 1.0F);

    for (int doc_id = 0; doc_id < kDocCount; ++doc_id) {
        std::map<int, float> doc;
        doc[1000] = value_dist(rng);
        doc[2000] = value_dist(rng);
        doc[3000] = value_dist(rng);
        doc[4000] = value_dist(rng);
        doc[doc_id] = value_dist(rng);
        docs.push_back(doc);
    }

    index.add_docs(docs);
    index.build();

    std::vector<idx_t> query_indptr = {0, 4};
    std::vector<term_t> query_indices = {1000, 2000, 3000, 4000};
    std::vector<float> query_values = {0.12F, 0.64F, 0.87F, 0.53F};

    std::vector<idx_t> labels_small(kDocCount, -1);
    std::vector<float> distances_small(kDocCount, -1.0F);
    SeismicSearchParameters params_small(2, 0.000001F);
    Index* idx = &index;
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), kDocCount, distances_small.data(),
                labels_small.data(), &params_small);

    int count_small = 0;
    for (int i = 0; i < kDocCount; ++i) {
        if (labels_small[i] >= 0 && labels_small[i] < kDocCount) {
            ++count_small;
        }
    }

    std::vector<idx_t> labels_large(kDocCount, -1);
    std::vector<float> distances_large(kDocCount, -1.0F);
    SeismicSearchParameters params_large(4, 100000.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), kDocCount, distances_large.data(),
                labels_large.data(), &params_large);

    int count_large = 0;
    for (int i = 0; i < kDocCount; ++i) {
        if (labels_large[i] >= 0 && labels_large[i] < kDocCount) {
            ++count_large;
        }
    }

    EXPECT_LE(count_small, kDocCount);
    EXPECT_GT(count_small, 0);
    EXPECT_EQ(count_large, kDocCount);
    EXPECT_LE(count_small, count_large);
}

// ============== IDSelector tests ==============

TEST(SeismicSQIndexSearch, search_exact_match_with_small_selector) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 3,
                                 0.5F, 3);
    Index* idx = &index;

    // doc0: term0=0.3, doc1: term0=1.0, doc2: term0=0.5, doc3: term0=0.8
    index.add_docs({{{0, 0.3F}}, {{0, 1.0F}}, {{0, 0.5F}}, {{0, 0.8F}}});
    index.build();

    // Selector size (2) <= k (2), triggers exact match path
    std::vector<idx_t> allowed_ids = {1, 3};
    ArrayIDSelector selector(allowed_ids.size(), allowed_ids.data());

    SeismicSearchParameters params(5, 1.0F);
    params.set_id_selector(&selector);

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(2, -1);
    std::vector<float> distances(2, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 2, distances.data(), labels.data(),
                &params);

    // Exact match should return doc1 and doc3, sorted by score descending
    EXPECT_EQ(labels[0], 1);
    EXPECT_EQ(labels[1], 3);
    EXPECT_GT(distances[0], distances[1]);
}

TEST(SeismicSQIndexSearch, search_exact_match_scores_match_normal_path_scores) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 3,
                                 0.5F, 3);
    Index* idx = &index;

    // doc0: term0=1.0, doc1: term0=0.5, doc2: term0=0.8
    index.add_docs({{{0, 1.0F}}, {{0, 0.5F}}, {{0, 0.8F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};

    // Normal path (no selector): k=3, all 3 docs returned
    std::vector<idx_t> labels_normal(3, -1);
    std::vector<float> distances_normal(3, -1.0F);
    SeismicSearchParameters params_normal(5, 1000.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 3, distances_normal.data(),
                labels_normal.data(), &params_normal);

    // Exact match path: selector size (2) <= k (2)
    std::vector<idx_t> allowed_ids = {0, 2};
    ArrayIDSelector selector(allowed_ids.size(), allowed_ids.data());
    SeismicSearchParameters params_filtered(5, 1000.0F);
    params_filtered.set_id_selector(&selector);

    std::vector<idx_t> labels_filtered(2, -1);
    std::vector<float> distances_filtered(2, -1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 2, distances_filtered.data(),
                labels_filtered.data(), &params_filtered);

    // Find doc0's score from the normal path
    float doc0_score_normal = -1.0F;
    for (int i = 0; i < 3; ++i) {
        if (labels_normal[i] == 0) {
            doc0_score_normal = distances_normal[i];
            break;
        }
    }
    // Find doc0's score from the exact match path
    float doc0_score_filtered = -1.0F;
    for (int i = 0; i < 2; ++i) {
        if (labels_filtered[i] == 0) {
            doc0_score_filtered = distances_filtered[i];
            break;
        }
    }

    ASSERT_GE(doc0_score_normal, 0.0F);
    ASSERT_GE(doc0_score_filtered, 0.0F);
    // Scores must match — both paths should decode quantized dot products
    EXPECT_FLOAT_EQ(doc0_score_normal, doc0_score_filtered);
    // Sanity: decoded score should be in a reasonable range (not raw quantized)
    EXPECT_LT(doc0_score_filtered, 10.0F);
}

TEST(SeismicSQIndexSearch, search_with_id_selector_filters_results) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 3,
                                 0.5F, 3);
    Index* idx = &index;

    // doc0: term0=0.3, doc1: term0=1.0, doc2: term0=0.5
    index.add_docs({{{0, 0.3F}}, {{0, 1.0F}}, {{0, 0.5F}}});
    index.build();

    // Only allow doc0 and doc2 via IDSelector
    std::vector<idx_t> allowed_ids = {0, 2};
    SetIDSelector selector(allowed_ids.size(), allowed_ids.data());

    SeismicSearchParameters params(5, 1.0F);
    params.set_id_selector(&selector);

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 3, distances.data(), labels.data(),
                &params);

    // doc1 (highest score) should be excluded
    // Results: doc2, doc0, then padding
    EXPECT_EQ(labels[0], 2);
    EXPECT_EQ(labels[1], 0);
    EXPECT_EQ(labels[2], -1);
}

// Parameters carrying no cut / heap factor fall back to the defaults, as they
// do for SeismicIndex, rather than being rejected or dereferenced as null.
TEST(SeismicSQIndexSearch, search_without_seismic_parameters_uses_defaults) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 3);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}, {{0, 0.2F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                nullptr);
    EXPECT_EQ(labels[0], 0);

    SearchParameters plain_params;
    labels[0] = -1;
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                &plain_params);
    EXPECT_EQ(labels[0], 0);
}

// ============== mapped read_index tests ==============

namespace {

// Index file removed on destruction.
class TempIndexFile {
public:
    explicit TempIndexFile(const std::string& name)
        : path_(std::filesystem::temp_directory_path() / name) {
        std::filesystem::remove(path_);
    }
    ~TempIndexFile() { std::filesystem::remove(path_); }

    TempIndexFile(const TempIndexFile&) = delete;
    TempIndexFile& operator=(const TempIndexFile&) = delete;

    // write_index/read_index take char*, not const char*.
    char* c_str() { return path_str_.data(); }

private:
    std::filesystem::path path_{};
    std::string path_str_ = path_.string();
};

// A built index with a handful of docs. Residency is a property of the reader,
// not of the file, so the written bytes are the same either way.
std::unique_ptr<TestableSeismicSQIndex> built_index(QuantizerType qtype) {
    auto index = std::make_unique<TestableSeismicSQIndex>(qtype, 0.0F, 1.0F, 10,
                                                          2, 0.5F, 5);
    index->add_docs({{{0, 1.0F}, {2, 0.9F}},
                     {{1, 0.5F}, {3, 0.7F}},
                     {{0, 0.3F}, {4, 0.8F}},
                     {{2, 0.6F}, {3, 0.4F}, {4, 0.2F}}});
    index->build();
    return index;
}

std::vector<idx_t> search_top(Index* index, term_t term, int k) {
    std::vector<idx_t> indptr = {0, 1};
    std::vector<term_t> indices = {term};
    std::vector<float> values = {1.0F};
    std::vector<idx_t> labels(k, INVALID_IDX);
    std::vector<float> distances(k, -1.0F);
    SeismicSearchParameters params(5, 1.0F);
    index->search(1, indptr.data(), indices.data(), values.data(), k,
                  distances.data(), labels.data(), &params);
    return labels;
}

}  // namespace

class SeismicSQIndexMmapIO : public testing::TestWithParam<QuantizerType> {};

INSTANTIATE_TEST_SUITE_P(QuantizerTypes, SeismicSQIndexMmapIO,
                         testing::Values(QuantizerType::QT_8bit,
                                         QuantizerType::QT_16bit),
                         [](const testing::TestParamInfo<QuantizerType>& info) {
                             return info.param == QuantizerType::QT_8bit
                                        ? "bit8"
                                        : "bit16";
                         });

TEST_P(SeismicSQIndexMmapIO, mapped_read_matches_the_copying_read) {
    TempIndexFile file("nsparse_sesq_mapped.idx");

    auto source = built_index(GetParam());
    write_index(source.get(), file.c_str());

    // One file, read both ways: kUseMmap is all that separates the residencies.
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    std::unique_ptr<Index> copied(read_index(file.c_str()));

    ASSERT_NE(mapped, nullptr);
    ASSERT_NE(copied, nullptr);
    ASSERT_EQ(mapped->get_vectors()->num_vectors(),
              copied->get_vectors()->num_vectors());
    // The quantizer header is part of the mapped payload, not just the copied
    // one: without it the codes would be decoded against the wrong range.
    const auto* mapped_sq =
        dynamic_cast<SeismicScalarQuantizedIndex*>(mapped.get());
    ASSERT_NE(mapped_sq, nullptr);
    EXPECT_EQ(mapped_sq->get_scalar_quantizer().get_quantizer_type(),
              GetParam());
    EXPECT_EQ(mapped->get_vectors()->get_element_size(),
              mapped_sq->get_scalar_quantizer().bytes_per_value());
    for (term_t term = 0; term < 5; ++term) {
        EXPECT_EQ(search_top(mapped.get(), term, 4),
                  search_top(copied.get(), term, 4))
            << "term " << term;
    }
}

// The point of the mapped path: the arrays point into the file, not into fresh
// allocations.
TEST(SeismicSQIndexMmapIOSingle, mapped_read_borrows_from_the_file) {
    TempIndexFile file("nsparse_sesq_borrow.idx");
    auto source = built_index(QuantizerType::QT_8bit);
    write_index(source.get(), file.c_str());

    std::unique_ptr<Index> loaded(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));

    const auto* vectors = loaded->get_vectors();
    ASSERT_NE(vectors, nullptr);
    // A borrowed CSR keeps the file's contiguity: indices sit right after
    // indptr, which is only true of the mapping, not of separate allocations.
    const auto* indptr_bytes =
        reinterpret_cast<const uint8_t*>(vectors->indptr_data());
    const auto* indices_bytes =
        reinterpret_cast<const uint8_t*>(vectors->indices_data());
    EXPECT_EQ(
        indices_bytes - indptr_bytes,
        static_cast<ptrdiff_t>((vectors->num_vectors() + 1) * sizeof(idx_t)));
}

// A stream has no file to map, so the flag alone must not send read_index down
// the mapped path.
TEST(SeismicSQIndexMmapIOSingle,
     buffered_read_copies_even_when_the_flag_is_set) {
    auto source = built_index(QuantizerType::QT_8bit);

    BufferedIOWriter writer;
    write_index(source.get(), &writer);

    BufferedIOReader reader(writer.data());
    std::unique_ptr<Index> loaded(read_index(&reader, IndexIoFlag::kUseMmap));

    ASSERT_NE(loaded, nullptr);
    const auto* vectors = loaded->get_vectors();
    ASSERT_NE(vectors, nullptr);
    EXPECT_EQ(vectors->num_vectors(), 4);
    // Copied out, so nothing points into the writer's buffer.
    const auto* base = writer.data().data();
    const auto* end = base + writer.data().size();
    const auto* indptr =
        reinterpret_cast<const uint8_t*>(vectors->indptr_data());
    EXPECT_TRUE(indptr < base || indptr >= end);
}

TEST(SeismicSQIndexMmapIOSingle, mapped_read_of_an_empty_index) {
    TempIndexFile file("nsparse_sesq_empty_mapped.idx");
    SeismicScalarQuantizedIndex source(QuantizerType::QT_16bit, 0.25F, 2.0F,
                                       {.lambda = 10, .beta = 2, .alpha = 0.5F},
                                       5);
    write_index(&source, file.c_str());

    std::unique_ptr<Index> loaded(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_vectors(), nullptr);
    const auto* loaded_sq =
        dynamic_cast<SeismicScalarQuantizedIndex*>(loaded.get());
    ASSERT_NE(loaded_sq, nullptr);
    EXPECT_EQ(loaded_sq->get_scalar_quantizer().get_quantizer_type(),
              QuantizerType::QT_16bit);
    EXPECT_FLOAT_EQ(loaded_sq->get_scalar_quantizer().get_min(), 0.25F);
    EXPECT_FLOAT_EQ(loaded_sq->get_scalar_quantizer().get_max(), 2.0F);
}

// A mapped CSR is borrowed at the width it was written in, which is float, so
// this index cannot take that path: its values have to be quantized by add().
TEST(SeismicSQIndexMmapIOSingle, read_csr_rejects_the_mapped_residency) {
    SeismicScalarQuantizedIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                      {.lambda = 10, .beta = 2, .alpha = 0.5F},
                                      5);
    EXPECT_THROW(index.read_csr("does_not_matter.csr", Residency::kMmap),
                 std::invalid_argument);
}

}  // namespace
}  // namespace nsparse
