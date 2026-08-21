# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Independent brute-force reference for accuracy checks.

Deliberately implemented in numpy rather than delegating to the ``brutal``
index: an in-library oracle would go green against its own regression.
"""

import numpy as np


def densify(csr, dim):
    """Expand a CSR triple into a dense (n_vectors, dim) float32 matrix."""
    indptr, indices, values = csr
    n = len(indptr) - 1
    dense = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        lo, hi = indptr[i], indptr[i + 1]
        dense[i, indices[lo:hi].astype(np.int64)] = values[lo:hi]
    return dense


def brute_force_top_k(docs_csr, queries_csr, dim, k):
    """Exact top-k by dot product, ranked score-desc then id-asc.

    Only positive scores are eligible -- the library returns INVALID_IDX
    padding rather than zero-scoring docs -- so short rows are padded with -1
    to match the shape the indexes return.
    """
    docs = densify(docs_csr, dim)
    queries = densify(queries_csr, dim)
    scores = queries @ docs.T

    labels = np.full((len(queries), k), -1, dtype=np.int64)
    dists = np.full((len(queries), k), -1.0, dtype=np.float32)
    for q in range(len(queries)):
        (eligible,) = np.nonzero(scores[q] > 0.0)
        # lexsort is stable and sorts by the last key first: score desc, id asc.
        order = eligible[np.lexsort((eligible, -scores[q][eligible]))][:k]
        labels[q, : len(order)] = order
        dists[q, : len(order)] = scores[q][order]
    return labels, dists


def recall_at_k(got_labels, want_labels):
    """Mean per-query overlap of returned ids with the oracle's ids.

    Set overlap, not sequence equality: ties are ordered by doc id inside the
    library, and asserting on that ordering would flake for equal scores.
    """
    per_query = []
    for got, want in zip(got_labels, want_labels):
        want_set = {int(i) for i in want if i >= 0}
        if not want_set:
            continue
        got_set = {int(i) for i in got if i >= 0}
        per_query.append(len(got_set & want_set) / len(want_set))
    return float(np.mean(per_query)) if per_query else 1.0


def assert_exact(got_labels, got_dists, want_labels, want_dists):
    """Assert an exact index reproduces the oracle ranking outright."""
    np.testing.assert_array_equal(got_labels, want_labels)
    np.testing.assert_allclose(got_dists, want_dists, rtol=1e-5, atol=1e-5)
