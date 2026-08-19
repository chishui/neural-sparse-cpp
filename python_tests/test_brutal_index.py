# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Black-box tests for the brutal (exhaustive) index.

Two lifecycle differences from the other indexes, both load-bearing here:
brutal has no build() step -- Index::build() is the base throw and brutal does
not override it -- and it cannot be serialised. Both are pinned in
test_error_paths.py.
"""

import numpy as np
import pytest

import nsparse
from oracle import assert_exact
from support import (
    K,
    PAD_DIST,
    PAD_LABEL,
    make_index,
    search,
    search_each,
    slice_corpus,
)

SPEC = "brutal"


@pytest.fixture(scope="module")
def index(corpus):
    return make_index(SPEC, corpus, needs_build=False)


def test_happy_case(corpus, queries, oracle):
    """factory -> ingest -> query -> accuracy. No build step, no mmap: brutal
    is exhaustive, so it must reproduce the oracle exactly rather than clear a
    recall floor."""
    index = make_index(SPEC, corpus, needs_build=False)
    assert index.num_vectors() == corpus.n
    assert index.get_dimension() == corpus.dim

    dists, labels = search(index, queries)
    assert labels.shape == (queries.n, K)
    want_labels, want_dists = oracle
    assert_exact(labels, dists, want_labels, want_dists)


def test_with_id_map(corpus, queries, oracle, doc_ids):
    """idmap over brutal returns caller ids in exactly the oracle's order."""
    index = make_index(f"idmap,{SPEC}", corpus, needs_build=False, ids=doc_ids)
    dists, labels = search(index, queries)

    want_labels, want_dists = oracle
    want_external = np.where(want_labels >= 0, doc_ids[want_labels], PAD_LABEL)
    np.testing.assert_array_equal(labels, want_external)
    np.testing.assert_allclose(dists, want_dists, rtol=1e-5, atol=1e-5)


def test_id_selector_is_ignored(index, queries, oracle):
    """Pins a fail-open gap: brutal accepts an id selector and ignores it.

    BrutalIndex::search takes a SearchParameters* but never reads
    get_id_selector(), so filtering silently does nothing here -- results come
    back unfiltered instead of restricted or rejected. Only seismic and
    seismic_sq implement filtering; idmap translates it for its delegate.
    Invert this test if brutal ever learns to filter.
    """
    allowed = np.arange(50, dtype=np.int32)
    selector = nsparse.SetIDSelector(allowed)
    params = nsparse.SearchParameters()
    params.set_id_selector(selector)

    _, labels = search(index, queries, params=params)
    want_labels, _ = oracle
    np.testing.assert_array_equal(labels, want_labels)
    assert not np.isin(labels[labels >= 0], allowed).all(), (
        "selector appears to be honoured now -- update this test"
    )


def test_incremental_add(corpus, queries, oracle):
    """Ingest split across add() calls matches a single-batch ingest."""
    index = nsparse.index_factory(corpus.dim, SPEC)
    bounds = [0, corpus.n // 3, 2 * corpus.n // 3, corpus.n]
    for lo, hi in zip(bounds, bounds[1:]):
        part = slice_corpus(corpus, lo, hi)
        index.add(part.n, part.indptr, part.indices, part.values)

    assert index.num_vectors() == corpus.n
    dists, labels = search(index, queries)
    want_labels, want_dists = oracle
    assert_exact(labels, dists, want_labels, want_dists)


def test_k_larger_than_corpus(corpus, queries):
    """Short result rows are padded, not truncated."""
    small = make_index(SPEC, slice_corpus(corpus, 0, 3), needs_build=False)
    dists, labels = search(small, queries, k=K)
    assert labels.shape == (queries.n, K)
    assert (labels[:, 3:] == PAD_LABEL).all()
    assert (dists[:, 3:] == PAD_DIST).all()


def test_batch_matches_single_query(index, queries):
    """Batched search is OpenMP-parallel over queries; it must agree with the
    one-at-a-time path exactly."""
    batch_d, batch_l = search(index, queries)
    single_d, single_l = search_each(index, queries)
    np.testing.assert_array_equal(batch_l, single_l)
    np.testing.assert_array_equal(batch_d, single_d)
