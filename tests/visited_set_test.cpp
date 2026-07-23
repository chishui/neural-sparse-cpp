/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/utils/visited_set.h"

#include <gtest/gtest.h>

namespace {

using nsparse::detail::VisitedSet;

// (a) Within one query an id is inserted once; repeats are deduped.
TEST(VisitedSetTest, dedup_within_query) {
    VisitedSet s(128);

    EXPECT_TRUE(s.insert(7));
    EXPECT_FALSE(s.insert(7));
    EXPECT_FALSE(s.insert(7));

    EXPECT_TRUE(s.insert(70));
    EXPECT_FALSE(s.insert(70));
}

// (b) new_query() fully resets: every id inserted last query is new again.
TEST(VisitedSetTest, new_query_resets_all_words) {
    VisitedSet s(256);

    // Dirty ids spread across several 64-bit words.
    for (size_t id : {0u, 63u, 64u, 130u, 255u}) {
        EXPECT_TRUE(s.insert(id));
        EXPECT_FALSE(s.insert(id));
    }

    s.new_query();

    for (size_t id : {0u, 63u, 64u, 130u, 255u}) {
        EXPECT_TRUE(s.insert(id)) << "id " << id << " not reset";
    }
}

// (c) Two ids in the same 64-bit word both dedup within a query and both clear
// on new_query, exercising the "touched once per word" reset invariant.
TEST(VisitedSetTest, two_ids_same_word_both_clear) {
    VisitedSet s(128);

    // 5 and 60 share word 0; the second insert must not re-push the word, yet
    // new_query must still clear both bits.
    EXPECT_TRUE(s.insert(5));
    EXPECT_TRUE(s.insert(60));
    EXPECT_FALSE(s.insert(5));
    EXPECT_FALSE(s.insert(60));

    s.new_query();

    EXPECT_TRUE(s.insert(5));
    EXPECT_TRUE(s.insert(60));
}

// (d) resize() clears touched_ so a later new_query does not iterate a stale
// word index (which after shrinking would be out of bounds on bits_).
TEST(VisitedSetTest, resize_clears_touched_and_bits) {
    VisitedSet s(256);
    EXPECT_TRUE(s.insert(200));  // dirties word 3

    s.resize(64);  // bits_ shrinks to a single word; touched_ must be cleared

    // If touched_ retained word 3, this would be an out-of-bounds write.
    s.new_query();

    // Bits are also cleared by resize: an id valid in the new domain is fresh.
    EXPECT_TRUE(s.insert(0));
    EXPECT_FALSE(s.insert(0));
}

// resize() to the same size still clears prior state.
TEST(VisitedSetTest, resize_same_size_clears_bits) {
    VisitedSet s(128);
    EXPECT_TRUE(s.insert(42));

    s.resize(128);

    EXPECT_TRUE(s.insert(42));
}

// (e) Word boundaries: bit 63 (top of word 0), bit 64 (bottom of word 1), and
// the last valid id n-1 all behave and reset independently.
TEST(VisitedSetTest, word_boundaries) {
    constexpr size_t n = 130;  // 3 words (192 bits); last id is 129
    VisitedSet s(n);

    EXPECT_TRUE(s.insert(63));
    EXPECT_TRUE(s.insert(64));
    EXPECT_TRUE(s.insert(n - 1));

    // Each is independent: neighbors are not marked by these inserts.
    EXPECT_TRUE(s.insert(62));
    EXPECT_TRUE(s.insert(65));
    EXPECT_TRUE(s.insert(n - 2));

    // And each dedups.
    EXPECT_FALSE(s.insert(63));
    EXPECT_FALSE(s.insert(64));
    EXPECT_FALSE(s.insert(n - 1));

    s.new_query();

    EXPECT_TRUE(s.insert(63));
    EXPECT_TRUE(s.insert(64));
    EXPECT_TRUE(s.insert(n - 1));
}

}  // namespace
