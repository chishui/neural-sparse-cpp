# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Performs one crashing operation, for test_error_paths.py.

Only the cases that take the interpreter down live here; everything that raises
is asserted in-process. See test_error_paths.py for why a wrong dtype on a
non-final buffer argument segfaults.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import nsparse
from support import make_corpus

DIM = 512

CASES = {}


def case(fn):
    CASES[fn.__name__] = fn
    return fn


def corpus():
    return make_corpus(200, DIM, 20, 0x11)


@case
def bad_indices_dtype():
    c = corpus()
    index = nsparse.index_factory(DIM, "inverted")
    index.add(c.n, c.indptr, c.indices.astype(np.int32), c.values)


@case
def bad_indptr_dtype():
    c = corpus()
    index = nsparse.index_factory(DIM, "inverted")
    index.add(c.n, c.indptr.astype(np.int64), c.indices, c.values)


if __name__ == "__main__":
    CASES[sys.argv[1]]()
    print("COMPLETED-WITHOUT-ERROR")
