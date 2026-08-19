# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""How the bindings behave when a caller gets it wrong.

C++ exceptions are translated, so these are ordinary in-process assertions:
std::invalid_argument arrives as ValueError, everything else derived from
std::exception as RuntimeError.

The dtype cases at the bottom still need a subprocess. A wrong dtype for indptr
or indices does not raise -- it segfaults. The typemaps do validate the buffer
format, but they call SWIG_fail after releasing their own Py_buffer, and the
fail: label then runs the freearg typemaps for the remaining arguments, whose
Py_buffer views were never initialised. So PyBuffer_Release() is handed
uninitialised stack memory. `values` is the last argument, so nothing follows it
and it fails cleanly -- which is exactly why only the other two crash. Fixable
by zero-initialising the views or guarding freearg on view.obj.
"""

from pathlib import Path

import numpy as np
import pytest

import nsparse
from conftest import run_isolated
from support import make_corpus, make_index, search

SIGSEGV = -11
PROBE = str(Path(__file__).parent / "_abort_probe.py")

DIM = 512


@pytest.fixture(scope="module")
def small_corpus():
    return make_corpus(200, DIM, 20, 0x11)


@pytest.fixture(scope="module")
def small_queries():
    return make_corpus(4, DIM, 6, 0x22)


@pytest.mark.parametrize(
    "description,message",
    [
        ("nonsense", "Unknown index type"),
        ("", "Description cannot be null or empty"),
        ("idmap", "idmap requires a delegate index type"),
    ],
)
def test_bad_index_spec_raises(description, message):
    with pytest.raises(ValueError, match=message):
        nsparse.index_factory(DIM, description)


def test_build_on_brutal_raises(small_corpus):
    """brutal has no build step: Index::build() is the base throw and brutal
    does not override it."""
    with pytest.raises(RuntimeError, match="not implemented"):
        make_index("brutal", small_corpus, needs_build=True)


def test_write_index_on_brutal_raises(small_corpus, tmp_path):
    """brutal is not serialisable."""
    index = make_index("brutal", small_corpus, needs_build=False)
    with pytest.raises(RuntimeError, match="does not support"):
        nsparse.write_index(index, str(tmp_path / "brutal.idx"))


def test_non_positive_k_raises(small_corpus, small_queries):
    index = make_index("inverted", small_corpus)
    with pytest.raises(ValueError, match="must be positive"):
        search(index, small_queries, k=0)


def test_term_id_out_of_range_raises(small_corpus):
    """Terms must be < dim. Rejected at build, not at add."""
    indices = small_corpus.indices.copy()
    indices[0] = DIM  # one past the last valid term
    index = nsparse.index_factory(DIM, "inverted")
    index.add(small_corpus.n, small_corpus.indptr, indices, small_corpus.values)
    with pytest.raises(ValueError, match="term_id out of range"):
        index.build()


def test_wrong_values_dtype_raises(small_corpus):
    """`values` is the last buffer argument, so its rejection is graceful."""
    index = nsparse.index_factory(DIM, "inverted")
    with pytest.raises(TypeError, match="float32"):
        index.add(
            small_corpus.n,
            small_corpus.indptr,
            small_corpus.indices,
            small_corpus.values.astype(np.float64),
        )


@pytest.mark.parametrize("case", ["bad_indptr_dtype", "bad_indices_dtype"])
def test_wrong_dtype_segfaults(case):
    """Pins the crash described in the module docstring.

    Rewrite as pytest.raises(TypeError) once the typemaps zero-initialise their
    Py_buffer views -- see test_wrong_values_dtype_raises for the target shape.
    """
    result = run_isolated([PROBE, case])
    assert "COMPLETED-WITHOUT-ERROR" not in result.stdout
    assert result.returncode == SIGSEGV, (
        f"{case}: expected SIGSEGV, got {result.returncode}\n{result.stderr}"
    )
