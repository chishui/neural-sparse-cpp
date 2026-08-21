# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Runs one build/query under a fixed OMP_NUM_THREADS, for test_threading.py.

A separate interpreter is required: the OpenMP runtime reads OMP_NUM_THREADS
when it initialises, so it cannot be varied from inside a running test. The
corpus is regenerated from seeds rather than shipped, so the parent and child
agree on the data without any file exchange.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import nsparse
from support import make_corpus, make_index, search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--load", help="load this index instead of building one")
    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-docs", type=int, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--doc-nnz", type=int, required=True)
    ap.add_argument("--corpus-seed", type=lambda s: int(s, 0), required=True)
    ap.add_argument("--n-queries", type=int, required=True)
    ap.add_argument("--query-nnz", type=int, required=True)
    ap.add_argument("--query-seed", type=lambda s: int(s, 0), required=True)
    args = ap.parse_args()

    corpus = make_corpus(args.n_docs, args.dim, args.doc_nnz, args.corpus_seed)
    queries = make_corpus(
        args.n_queries, args.dim, args.query_nnz, args.query_seed
    )

    t0 = time.perf_counter()
    if args.load:
        index = nsparse.read_index(
            args.load, nsparse.kUseMmap if args.mmap else 0
        )
    else:
        index = make_index(args.spec, corpus, needs_build=not args.no_build)
    t_build = time.perf_counter() - t0

    params = None
    if args.spec.startswith("seismic_sq"):
        # The quantized index rejects null search parameters.
        params = nsparse.SeismicSQSearchParameters(0.0, 1.0, args.k, 1.2)

    t0 = time.perf_counter()
    dists, labels = search(index, queries, k=args.k, params=params)
    t_search = time.perf_counter() - t0

    np.savez(args.out, labels=labels, dists=dists)
    # Echoed so the parent can confirm the setting reached this process.
    print(
        f"omp_num_threads={os.environ.get('OMP_NUM_THREADS', 'unset')} "
        f"load_or_build={t_build:.3f}s search={t_search:.3f}s"
    )


if __name__ == "__main__":
    main()
