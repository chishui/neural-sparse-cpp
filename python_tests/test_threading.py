# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Thread-count behaviour of build and query.

Both build (over inverted lists) and search (over queries) are OpenMP-parallel.
What can be asserted differs between them:

* Search over an already-built index has no randomness, and each query writes
  to its own output slot, so results must be *bit-identical* at any thread
  count. Any difference means the per-thread scratch buffers in the search loop
  are leaking state between the queries a thread handles.

* Build cannot be asserted that way. RandomKMeans seeds itself from
  std::random_device with no seed parameter, so two builds of the same corpus
  differ even single-threaded. Only the recall floor is checkable.

Everything runs through _thread_worker.py in a fresh interpreter, because
OMP_NUM_THREADS is read when the OpenMP runtime initialises.

Note the limitation: these tests prove results do not *change* with the
requested thread count, and the worker echoes the value it received, but
nothing here proves OpenMP actually spawned that many threads. The worker's
timings are printed for that (visible with -s).
"""

import numpy as np
import pytest

import nsparse
from conftest import (
    CORPUS_SEED,
    DIM,
    DOC_NNZ,
    DOC_NNZ as _DOC_NNZ,
    N_DOCS,
    QUERY_NNZ,
    QUERY_SEED,
    run_isolated,
)
from oracle import recall_at_k
from support import K, make_index

THREAD_COUNTS = [1, 2, 4, 8]

# More queries than the shared fixture uses, so the parallel-over-queries loop
# has enough work for several threads to actually participate.
N_QUERIES = 400

# Persistable specs only: brutal cannot be serialised, so it cannot be built
# once and reloaded per thread count.
QUERY_SPECS = [
    "inverted",
    "seismic,lambda=25|beta=4|alpha=0.4",
    "seismic_sq,quantizer=8bit|vmin=0.0|vmax=1.0|lambda=25|beta=4|alpha=0.4",
]

BUILD_RECALL_FLOOR = 0.75


def _worker_args(spec, out, **extra):
    args = [
        "--spec", spec,
        "--out", str(out),
        "--k", str(K),
        "--n-docs", str(N_DOCS),
        "--dim", str(DIM),
        "--doc-nnz", str(DOC_NNZ),
        "--corpus-seed", hex(CORPUS_SEED),
        "--n-queries", str(N_QUERIES),
        "--query-nnz", str(QUERY_NNZ),
        "--query-seed", hex(QUERY_SEED),
    ]
    for key, value in extra.items():
        flag = "--" + key.replace("_", "-")
        args += [flag] if value is True else [flag, str(value)]
    return args


def _run(worker_script, nthreads, spec, out, capsys=None, **extra):
    result = run_isolated(
        [worker_script, *_worker_args(spec, out, **extra)],
        env={"OMP_NUM_THREADS": str(nthreads)},
    )
    assert result.returncode == 0, (
        f"worker failed at OMP_NUM_THREADS={nthreads}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert f"omp_num_threads={nthreads}" in result.stdout
    print(f"  [threads={nthreads}] {result.stdout.strip()}")
    with np.load(out) as data:
        return data["labels"], data["dists"]


@pytest.fixture(scope="module", params=QUERY_SPECS)
def prebuilt(request, corpus, tmp_path_factory):
    """One index, built once and persisted, so every thread count queries the
    exact same structure -- isolating search from the unseeded build RNG."""
    spec = request.param
    index = make_index(spec, corpus)
    path = tmp_path_factory.mktemp("threading") / "index.idx"
    nsparse.write_index(index, str(path))
    return spec, path


@pytest.mark.parametrize("nthreads", THREAD_COUNTS[1:])
def test_query_thread_count_invariance(
    prebuilt, nthreads, worker_script, tmp_path
):
    """Querying one fixed index must give identical results at any thread
    count."""
    spec, index_path = prebuilt
    base_l, base_d = _run(
        worker_script, 1, spec, tmp_path / "t1.npz", load=str(index_path)
    )
    got_l, got_d = _run(
        worker_script,
        nthreads,
        spec,
        tmp_path / f"t{nthreads}.npz",
        load=str(index_path),
    )
    np.testing.assert_array_equal(got_l, base_l)
    np.testing.assert_array_equal(got_d, base_d)


@pytest.mark.parametrize("nthreads", THREAD_COUNTS[1:])
def test_query_thread_count_invariance_mmap(
    prebuilt, nthreads, worker_script, tmp_path
):
    """Same, with the index mapped rather than read into memory."""
    spec, index_path = prebuilt
    base_l, base_d = _run(
        worker_script, 1, spec, tmp_path / "m1.npz",
        load=str(index_path), mmap=True,
    )
    got_l, got_d = _run(
        worker_script, nthreads, spec, tmp_path / f"m{nthreads}.npz",
        load=str(index_path), mmap=True,
    )
    np.testing.assert_array_equal(got_l, base_l)
    np.testing.assert_array_equal(got_d, base_d)


@pytest.mark.parametrize("nthreads", THREAD_COUNTS)
def test_build_thread_count_recall(nthreads, worker_script, tmp_path, corpus):
    """A parallel build must produce an index of the same quality.

    Only a recall floor, not equality: RandomKMeans is seeded from
    std::random_device, so builds are not reproducible at any thread count.
    """
    from oracle import brute_force_top_k
    from support import make_corpus

    spec = "seismic,lambda=25|beta=4|alpha=0.4"
    got_l, _ = _run(
        worker_script, nthreads, spec, tmp_path / f"b{nthreads}.npz"
    )
    queries = make_corpus(N_QUERIES, DIM, QUERY_NNZ, QUERY_SEED)
    want_l, _ = brute_force_top_k(corpus.csr, queries.csr, DIM, K)
    assert recall_at_k(got_l, want_l) >= BUILD_RECALL_FLOOR
