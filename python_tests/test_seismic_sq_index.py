# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Black-box tests for the scalar-quantized seismic index.

Every query here passes SearchParameters explicitly, which is required for the
file to work on both sides of a behaviour change: on main,
SeismicScalarQuantizedIndex::search starts with
throw_if_null(search_parameters, ...), whereas the mmap-residency work relaxed
that so a null reads the index's own quantizer. Passing params is valid either
way.

Mapped residency for this index type is also newer than main, so the mmap case
is skipped where the class is not an MmapIndex.
"""

import numpy as np
import pytest

import nsparse
from oracle import recall_at_k
from support import (
    K,
    PAD_DIST,
    PAD_LABEL,
    make_index,
    roundtrip,
    search,
    search_each,
    slice_corpus,
)

VMIN, VMAX = 0.0, 1.0
CUT, HEAP_FACTOR = K, 1.2
SPEC = (
    f"seismic_sq,quantizer=8bit|vmin={VMIN}|vmax={VMAX}"
    "|lambda=25|beta=4|alpha=0.4"
)

# A floor with headroom, not a target: the build RNG is unseeded, so recall
# moves from run to run and a tight bound would flake.
RECALL_FLOOR = 0.82


def params(cut=CUT, heap_factor=HEAP_FACTOR):
    return nsparse.SeismicSQSearchParameters(VMIN, VMAX, cut, heap_factor)


@pytest.fixture(scope="module")
def index(corpus):
    return make_index(SPEC, corpus)


@pytest.mark.parametrize("residency", ["memory", "mmap"])
def test_happy_case(residency, corpus, queries, oracle, tmp_path):
    """factory -> ingest -> build -> query -> accuracy, in both residencies."""
    index = make_index(SPEC, corpus)
    assert index.num_vectors() == corpus.n
    assert index.get_dimension() == corpus.dim

    if residency == "mmap":
        if not issubclass(
            nsparse.SeismicScalarQuantizedIndex, nsparse.MmapIndex
        ):
            pytest.skip("this build has no mapped residency for seismic_sq")
        index = roundtrip(index, tmp_path / "seismic_sq.idx", nsparse.kUseMmap)

    dists, labels = search(index, queries, params=params())
    assert labels.shape == (queries.n, K)
    assert dists.shape == (queries.n, K)
    assert (labels[:, 0] >= 0).all()

    want_labels, _ = oracle
    assert recall_at_k(labels, want_labels) >= RECALL_FLOOR


def test_persistence_roundtrip(index, queries, tmp_path):
    """Reloading reproduces the results of the index that wrote the file."""
    before_d, before_l = search(index, queries, params=params())
    reloaded = roundtrip(index, tmp_path / "seismic_sq.idx")
    after_d, after_l = search(reloaded, queries, params=params())
    np.testing.assert_array_equal(after_l, before_l)
    np.testing.assert_allclose(after_d, before_d, rtol=1e-6, atol=1e-6)


def test_with_id_map(corpus, queries, oracle, doc_ids):
    """idmap returns the caller's ids, not internal ordinals."""
    index = make_index(f"idmap,{SPEC}", corpus, ids=doc_ids)
    _, labels = search(index, queries, params=params())

    returned = labels[labels >= 0]
    assert np.isin(returned, doc_ids).all()

    want_labels, _ = oracle
    want_external = np.where(want_labels >= 0, doc_ids[want_labels], PAD_LABEL)
    assert recall_at_k(labels, want_external) >= RECALL_FLOOR


def test_exact_match(index, queries, oracle):
    """An enumerable selector of size <= k switches to the exact path.

    Scores still come from the quantized values, so only the id set is asserted
    exactly; the scores are compared with a quantization-sized tolerance.
    """
    want_labels, want_dists = oracle
    ids = np.ascontiguousarray(
        want_labels[0][want_labels[0] >= 0], dtype=np.int32
    )
    assert len(ids) == K

    selector = nsparse.SetIDSelector(ids)
    p = params()
    p.set_id_selector(selector)

    dists, labels = _search_one(index, queries, 0, p)
    assert set(int(i) for i in labels) == set(int(i) for i in ids)
    # 8-bit over [vmin, vmax] gives a step of (vmax-vmin)/255 per component;
    # with QUERY_NNZ terms the accumulated error stays well inside 0.1.
    np.testing.assert_allclose(np.sort(dists), np.sort(want_dists[0]), atol=0.1)


def test_filtered_search(index, queries, oracle):
    """A selector larger than k filters but stays on the approximate path."""
    want_labels, _ = oracle
    allowed = np.ascontiguousarray(
        np.unique(want_labels[want_labels >= 0])[: K * 5], dtype=np.int32
    )
    assert len(allowed) > K

    selector = nsparse.SetIDSelector(allowed)
    p = params()
    p.set_id_selector(selector)

    _, labels = search(index, queries, params=p)
    assert np.isin(labels[labels >= 0], allowed).all()


def test_excluded_ids(index, queries, oracle):
    """NotIDSelector removes ids; not enumerable, so no exact path."""
    want_labels, _ = oracle
    banned = np.ascontiguousarray(
        want_labels[0][:5][want_labels[0][:5] >= 0], dtype=np.int32
    )
    # Held in locals: NotIDSelector keeps only a raw pointer to its delegate.
    inner = nsparse.SetIDSelector(banned)
    selector = nsparse.NotIDSelector(inner)
    p = params()
    p.set_id_selector(selector)

    _, labels = _search_one(index, queries, 0, p)
    assert not np.isin(labels[labels >= 0], banned).any()


def test_incremental_add(corpus, queries, oracle):
    """Ingest split across add() calls matches a single-batch ingest."""
    index = nsparse.index_factory(corpus.dim, SPEC)
    bounds = [0, corpus.n // 3, 2 * corpus.n // 3, corpus.n]
    for lo, hi in zip(bounds, bounds[1:]):
        part = slice_corpus(corpus, lo, hi)
        index.add(part.n, part.indptr, part.indices, part.values)
    index.build()

    assert index.num_vectors() == corpus.n
    _, labels = search(index, queries, params=params())
    want_labels, _ = oracle
    assert recall_at_k(labels, want_labels) >= RECALL_FLOOR


def test_k_larger_than_corpus(corpus, queries):
    """Short result rows are padded, not truncated."""
    small = make_index(SPEC, slice_corpus(corpus, 0, 3))
    dists, labels = search(small, queries, k=K, params=params())
    assert labels.shape == (queries.n, K)
    assert (labels[:, 3:] == PAD_LABEL).all()
    assert (dists[:, 3:] == PAD_DIST).all()


def test_batch_matches_single_query(index, queries):
    """Batched search is OpenMP-parallel over queries; per-thread scratch is
    reused across queries, so batched must equal one-at-a-time exactly."""
    batch_d, batch_l = search(index, queries, params=params())
    single_d, single_l = search_each(index, queries, params=params())
    np.testing.assert_array_equal(batch_l, single_l)
    np.testing.assert_array_equal(batch_d, single_d)


@pytest.mark.parametrize("width", ["8bit", "16bit"])
def test_quantizer_widths(corpus, queries, oracle, width):
    """Both quantizer widths clear the recall floor.

    Not asserted: that 16bit beats 8bit. It does on average, but the unseeded
    build RNG spreads each width's recall far enough that their ranges overlap,
    so a per-run comparison would flake.
    """
    spec = (
        f"seismic_sq,quantizer={width}|vmin={VMIN}|vmax={VMAX}"
        "|lambda=25|beta=4|alpha=0.4"
    )
    index = make_index(spec, corpus)
    _, labels = search(index, queries, params=params())
    want_labels, _ = oracle
    assert recall_at_k(labels, want_labels) >= RECALL_FLOOR


def test_quantization_range(corpus, queries, oracle):
    """A range that clips the data still returns k well-formed hits.

    The corpus values span [0.1, 1.0]; vmax=0.5 clips the top half, which
    should cost recall without breaking the result contract.
    """
    spec = (
        f"seismic_sq,quantizer=8bit|vmin=0.0|vmax=0.5"
        "|lambda=25|beta=4|alpha=0.4"
    )
    index = make_index(spec, corpus)
    p = nsparse.SeismicSQSearchParameters(0.0, 0.5, CUT, HEAP_FACTOR)
    _, labels = search(index, queries, params=p)
    assert labels.shape == (queries.n, K)
    assert (labels[:, 0] >= 0).all()
    want_labels, _ = oracle
    assert recall_at_k(labels, want_labels) > 0.0


# --- helpers -------------------------------------------------------------


def _search_one(index, queries, q, p, k=K):
    lo, hi = queries.indptr[q], queries.indptr[q + 1]
    d, l = index.search(
        1,
        np.array([0, hi - lo], dtype=np.int32),
        np.ascontiguousarray(queries.indices[lo:hi]),
        np.ascontiguousarray(queries.values[lo:hi]),
        k,
        p,
    )
    return d[0], l[0]
