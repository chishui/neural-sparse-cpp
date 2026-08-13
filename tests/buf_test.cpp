/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/utils/buf.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nsparse/types.h"

namespace {

// Counts releases, so tests can assert one ran exactly once and at the right
// time. Reset before each use.
int g_release_count = 0;
void* g_last_context = nullptr;

void counting_release(void* context) {
    ++g_release_count;
    g_last_context = context;
}

// A second releaser, to check that context_as() keys on identity.
void other_release(void* /*context*/) {}

void reset_release_counters() {
    g_release_count = 0;
    g_last_context = nullptr;
}

}  // namespace

// MemoryOwner

TEST(MemoryOwner, default_owns_nothing) {
    nsparse::MemoryOwner owner;
    ASSERT_FALSE(owner.owns());
    ASSERT_EQ(owner.context(), nullptr);
    ASSERT_EQ(owner.releaser(), &nsparse::release_nothing);
}

TEST(MemoryOwner, runs_releaser_on_destruction) {
    reset_release_counters();
    int context = 7;
    {
        nsparse::MemoryOwner owner(&context, &counting_release);
        ASSERT_TRUE(owner.owns());
        ASSERT_EQ(g_release_count, 0);
    }
    ASSERT_EQ(g_release_count, 1);
    ASSERT_EQ(g_last_context, &context);
}

TEST(MemoryOwner, null_releaser_is_non_owning) {
    int context = 7;
    nsparse::MemoryOwner owner(&context, nullptr);
    ASSERT_FALSE(owner.owns());
    // The context is still reachable; only the release is suppressed.
    ASSERT_EQ(owner.context(), &context);
}

// A non-null context with a no-op releaser must not read as owning, or callers
// cannot tell a borrow from a release-on-destroy.
TEST(MemoryOwner, release_nothing_is_non_owning) {
    int context = 7;
    nsparse::MemoryOwner owner(&context, &nsparse::release_nothing);
    ASSERT_FALSE(owner.owns());
}

TEST(MemoryOwner, move_transfers_ownership_without_releasing) {
    reset_release_counters();
    int context = 7;
    {
        nsparse::MemoryOwner source(&context, &counting_release);
        nsparse::MemoryOwner sink(std::move(source));
        ASSERT_TRUE(sink.owns());
        ASSERT_EQ(sink.context(), &context);
        ASSERT_EQ(g_release_count, 0);
    }
    // Released once by the sink, not twice.
    ASSERT_EQ(g_release_count, 1);
}

TEST(MemoryOwner, move_assignment_releases_the_previous_context) {
    reset_release_counters();
    int first = 1;
    int second = 2;
    nsparse::MemoryOwner sink(&first, &counting_release);
    sink = nsparse::MemoryOwner(&second, &counting_release);
    ASSERT_EQ(g_release_count, 1);
    ASSERT_EQ(g_last_context, &first);
    ASSERT_EQ(sink.context(), &second);
}

TEST(MemoryOwner, context_as_requires_the_matching_releaser) {
    int context = 7;
    nsparse::MemoryOwner owner(&context, &counting_release);
    ASSERT_EQ(owner.context_as<int>(&counting_release), &context);
    ASSERT_EQ(owner.context_as<int>(&other_release), nullptr);
}

TEST(MemoryOwner, release_suppresses_the_releaser) {
    reset_release_counters();
    int context = 7;
    {
        nsparse::MemoryOwner owner(&context, &counting_release);
        ASSERT_EQ(owner.release(), &context);
    }
    ASSERT_EQ(g_release_count, 0);
}

// Buf: owning

TEST(Buf, default_is_empty_and_non_owning) {
    nsparse::Buf<float> buf;
    ASSERT_EQ(buf.data(), nullptr);
    ASSERT_EQ(buf.size(), 0);
    ASSERT_EQ(buf.byte_size(), 0);
    ASSERT_TRUE(buf.empty());
    ASSERT_FALSE(buf.owns());
}

TEST(Buf, own_adopts_the_vector_buffer_without_copying) {
    std::vector<float> values = {1.0F, 2.0F, 3.0F, 4.0F};
    const float* address = values.data();

    auto buf = nsparse::Buf<float>::own(std::move(values));

    ASSERT_EQ(buf.data(), address);
    ASSERT_EQ(buf.size(), 4);
    ASSERT_EQ(buf.byte_size(), 4 * sizeof(float));
    ASSERT_TRUE(buf.owns());
    ASSERT_FLOAT_EQ(buf[0], 1.0F);
    ASSERT_FLOAT_EQ(buf[3], 4.0F);
}

TEST(Buf, own_empty_vector) {
    auto buf = nsparse::Buf<nsparse::idx_t>::own(std::vector<nsparse::idx_t>{});
    ASSERT_EQ(buf.size(), 0);
    ASSERT_TRUE(buf.empty());
    // Still owning: the (empty) vector has to be destroyed.
    ASSERT_TRUE(buf.owns());
}

TEST(Buf, owned_vector_recovers_the_staging_vector) {
    std::vector<nsparse::term_t> values = {5, 6, 7};
    auto buf = nsparse::Buf<nsparse::term_t>::own(std::move(values));

    const auto* recovered = buf.owned_vector();
    ASSERT_NE(recovered, nullptr);
    ASSERT_EQ(recovered->size(), 3);
    ASSERT_EQ(recovered->data(), buf.data());
}

// Buf: take_vector (the append-style build path)

TEST(Buf, take_vector_moves_the_buffer_out_and_empties_the_buf) {
    std::vector<nsparse::idx_t> values = {1, 2, 3};
    const nsparse::idx_t* address = values.data();
    auto buf = nsparse::Buf<nsparse::idx_t>::own(std::move(values));

    std::vector<nsparse::idx_t> taken = buf.take_vector();

    ASSERT_EQ(taken.size(), 3);
    ASSERT_EQ(taken.data(), address);  // moved, not copied
    ASSERT_EQ(buf.size(), 0);
    ASSERT_EQ(buf.data(), nullptr);
    ASSERT_FALSE(buf.owns());
}

// A first append has nothing to take, so the default state must not throw.
TEST(Buf, take_vector_on_default_buf_yields_empty) {
    nsparse::Buf<nsparse::idx_t> buf;
    std::vector<nsparse::idx_t> taken = buf.take_vector();
    ASSERT_TRUE(taken.empty());
}

TEST(Buf, take_vector_round_trip_grows_and_preserves_contents) {
    nsparse::Buf<nsparse::idx_t> buf;
    for (nsparse::idx_t i = 0; i < 6; ++i) {
        std::vector<nsparse::idx_t> staging = buf.take_vector();
        staging.push_back(i);
        buf = nsparse::Buf<nsparse::idx_t>::own(std::move(staging));
    }
    ASSERT_EQ(buf.size(), 6);
    for (nsparse::idx_t i = 0; i < 6; ++i) {
        ASSERT_EQ(buf[i], i);
    }
    ASSERT_TRUE(buf.owns());
}

// Capacity must survive the round trip, or every append recopies the buffer.
TEST(Buf, take_vector_preserves_capacity) {
    std::vector<nsparse::idx_t> values;
    values.reserve(64);
    values.push_back(1);
    auto buf = nsparse::Buf<nsparse::idx_t>::own(std::move(values));

    std::vector<nsparse::idx_t> taken = buf.take_vector();
    ASSERT_GE(taken.capacity(), 64);
}

// Growing a borrowed or adopted buffer would write to memory it does not own.
TEST(Buf, take_vector_rejects_a_borrowed_buffer) {
    std::vector<nsparse::idx_t> values = {1, 2, 3};
    auto buf = nsparse::Buf<nsparse::idx_t>::borrow(values.data(), values.size());
    ASSERT_THROW(static_cast<void>(buf.take_vector()), std::runtime_error);
}

TEST(Buf, take_vector_rejects_an_adopted_buffer) {
    reset_release_counters();
    std::vector<nsparse::idx_t> region = {1, 2, 3, 4};
    int mapping_handle = 0;
    auto buf = nsparse::Buf<nsparse::idx_t>::adopt(region.data(), region.size(),
                                                   &mapping_handle,
                                                   &counting_release);
    ASSERT_TRUE(buf.owns());  // owning, but not a vector we allocated
    ASSERT_THROW(static_cast<void>(buf.take_vector()), std::runtime_error);
}

// Buf: borrowing

TEST(Buf, borrow_points_at_caller_memory) {
    std::vector<float> values = {1.0F, 2.0F, 3.0F};
    auto buf = nsparse::Buf<float>::borrow(values.data(), values.size());

    ASSERT_EQ(buf.data(), values.data());
    ASSERT_EQ(buf.size(), 3);
    ASSERT_FALSE(buf.owns());
    // No staging vector to recover, even though the memory came from one.
    ASSERT_EQ(buf.owned_vector(), nullptr);
}

// The property the mmap path depends on.
TEST(Buf, borrow_does_not_release_on_destruction) {
    reset_release_counters();
    std::vector<float> values = {1.0F, 2.0F, 3.0F};
    {
        auto buf = nsparse::Buf<float>::borrow(values.data(), values.size());
        ASSERT_FALSE(buf.owns());
    }
    ASSERT_EQ(g_release_count, 0);
    ASSERT_FLOAT_EQ(values[0], 1.0F);
}

TEST(Buf, borrow_sees_writes_through_the_original_buffer) {
    std::vector<float> values = {1.0F, 2.0F};
    auto buf = nsparse::Buf<float>::borrow(values.data(), values.size());
    values[1] = 9.0F;
    ASSERT_FLOAT_EQ(buf[1], 9.0F);
}

// Buf: adopting a foreign resource

TEST(Buf, adopt_releases_the_context_not_the_data) {
    reset_release_counters();
    // Stands in for a mapping: `region` is released whole, while the Buf
    // points at an interior offset.
    std::vector<uint8_t> region(64, 0);
    region[16] = 42;
    int mapping_handle = 0;
    {
        auto buf = nsparse::Buf<uint8_t>::adopt(region.data() + 16, 8,
                                                &mapping_handle,
                                                &counting_release);
        ASSERT_TRUE(buf.owns());
        ASSERT_EQ(buf.data(), region.data() + 16);
        ASSERT_EQ(buf.size(), 8);
        ASSERT_EQ(buf[0], 42);
        ASSERT_EQ(g_release_count, 0);
    }
    ASSERT_EQ(g_release_count, 1);
    ASSERT_EQ(g_last_context, &mapping_handle);
}

// Each buffer holds its own reference, so the region outlives the last one.
TEST(Buf, several_adopted_buffers_can_share_one_region) {
    reset_release_counters();
    std::vector<uint8_t> region(64, 7);
    auto shared = std::make_shared<int>(0);

    auto releaser = [](void* context) {
        counting_release(context);
        delete static_cast<std::shared_ptr<int>*>(context);
    };

    {
        auto first = nsparse::Buf<uint8_t>::adopt(
            region.data(), 32, new std::shared_ptr<int>(shared), releaser);
        ASSERT_EQ(shared.use_count(), 2);
        {
            auto second = nsparse::Buf<uint8_t>::adopt(
                region.data() + 32, 32, new std::shared_ptr<int>(shared),
                releaser);
            ASSERT_EQ(shared.use_count(), 3);
            ASSERT_EQ(second.size(), 32);
        }
        ASSERT_EQ(g_release_count, 1);
        ASSERT_EQ(shared.use_count(), 2);
    }
    ASSERT_EQ(g_release_count, 2);
    ASSERT_EQ(shared.use_count(), 1);
}

TEST(Buf, adopt_with_null_releaser_is_borrowing) {
    std::vector<float> values = {1.0F};
    int context = 0;
    auto buf =
        nsparse::Buf<float>::adopt(values.data(), values.size(), &context,
                                   nullptr);
    ASSERT_FALSE(buf.owns());
}

// Buf: access

TEST(Buf, at_bounds_checks) {
    auto buf = nsparse::Buf<nsparse::idx_t>::own(
        std::vector<nsparse::idx_t>{10, 20, 30});
    ASSERT_EQ(buf.at(2), 30);
    ASSERT_THROW(static_cast<void>(buf.at(3)), std::out_of_range);
}

TEST(Buf, iteration_covers_owned_and_borrowed_alike) {
    std::vector<nsparse::idx_t> values(5);
    std::iota(values.begin(), values.end(), 1);
    const nsparse::idx_t expected = 1 + 2 + 3 + 4 + 5;

    auto borrowed =
        nsparse::Buf<nsparse::idx_t>::borrow(values.data(), values.size());
    nsparse::idx_t borrowed_sum = 0;
    for (const auto& value : borrowed) {
        borrowed_sum += value;
    }

    auto owned = nsparse::Buf<nsparse::idx_t>::own(std::move(values));
    nsparse::idx_t owned_sum = 0;
    for (const auto& value : owned) {
        owned_sum += value;
    }

    ASSERT_EQ(borrowed_sum, expected);
    ASSERT_EQ(owned_sum, expected);
}

// Buf: move semantics
//
// Buf members live in containers that relocate during build and deserialize; a
// throwing move would silently degrade those to copies.
TEST(Buf, move_is_noexcept) {
    ASSERT_TRUE(std::is_nothrow_move_constructible_v<nsparse::Buf<float>>);
    ASSERT_TRUE(std::is_nothrow_move_assignable_v<nsparse::Buf<float>>);
}

TEST(Buf, move_transfers_without_releasing) {
    reset_release_counters();
    std::vector<float> values = {1.0F, 2.0F};
    const float* address = values.data();
    {
        auto source = nsparse::Buf<float>::own(std::move(values));
        auto sink = std::move(source);
        ASSERT_EQ(sink.data(), address);
        ASSERT_EQ(sink.size(), 2);
        ASSERT_TRUE(sink.owns());
    }
    ASSERT_EQ(g_release_count, 0);  // released the vector, not our counter
}

TEST(Buf, move_assignment_releases_the_previous_buffer) {
    reset_release_counters();
    int first = 1;
    int second = 2;
    std::vector<float> data(4, 0.0F);

    auto sink = nsparse::Buf<float>::adopt(data.data(), 2, &first,
                                           &counting_release);
    sink = nsparse::Buf<float>::adopt(data.data() + 2, 2, &second,
                                      &counting_release);

    ASSERT_EQ(g_release_count, 1);
    ASSERT_EQ(g_last_context, &first);
    ASSERT_EQ(sink.data(), data.data() + 2);
}

TEST(Buf, survives_relocation_inside_a_vector) {
    std::vector<nsparse::Buf<nsparse::idx_t>> buffers;
    for (nsparse::idx_t i = 0; i < 8; ++i) {
        buffers.push_back(nsparse::Buf<nsparse::idx_t>::own(
            std::vector<nsparse::idx_t>{i, i, i}));
    }
    ASSERT_EQ(buffers.size(), 8);
    for (nsparse::idx_t i = 0; i < 8; ++i) {
        ASSERT_EQ(buffers[i].size(), 3);
        ASSERT_EQ(buffers[i][0], i);
        ASSERT_TRUE(buffers[i].owns());
    }
}

// Buf: footprint
//
// One Buf per array is only affordable while a Buf stays small; pinned so a new
// member cannot silently inflate every array.
TEST(Buf, footprint_is_pointer_size_owner) {
    ASSERT_EQ(sizeof(nsparse::MemoryOwner), 2 * sizeof(void*));
    ASSERT_EQ(sizeof(nsparse::Buf<float>),
              sizeof(void*) + sizeof(size_t) + sizeof(nsparse::MemoryOwner));
}
