import numpy as np
from syqma.bitint import (
    pack_array_to_int,
    unpack_int_to_array,
    pack_rows_to_ints,
    unpack_batch_of_ints_to_array,
    unpack_batch_of_ints_to_array_fast,
)


class TestPackUnpackSingle:
    def test_roundtrip_simple(self):
        x = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        packed = pack_array_to_int(x)
        unpacked = unpack_int_to_array(packed, len(x))
        assert np.array_equal(unpacked, x)

    def test_roundtrip_all_zeros(self):
        x = np.zeros(8, dtype=np.uint8)
        packed = pack_array_to_int(x)
        assert packed == 0
        unpacked = unpack_int_to_array(packed, 8)
        assert np.array_equal(unpacked, x)

    def test_roundtrip_all_ones(self):
        x = np.ones(8, dtype=np.uint8)
        packed = pack_array_to_int(x)
        assert packed == 0xFF
        unpacked = unpack_int_to_array(packed, 8)
        assert np.array_equal(unpacked, x)

    def test_single_bit(self):
        for bit in [0, 1]:
            x = np.array([bit], dtype=np.uint8)
            packed = pack_array_to_int(x)
            unpacked = unpack_int_to_array(packed, 1)
            assert np.array_equal(unpacked, x)

    def test_lsb_ordering(self):
        # [1, 0, 0, 0] with LSB packing => integer 1
        x = np.array([1, 0, 0, 0], dtype=np.uint8)
        packed = pack_array_to_int(x)
        assert packed == 1

    def test_larger_array(self):
        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, size=100, dtype=np.uint8)
        packed = pack_array_to_int(x)
        unpacked = unpack_int_to_array(packed, 100)
        assert np.array_equal(unpacked, x)


class TestPackUnpackBatch:
    def test_roundtrip(self):
        rng = np.random.default_rng(123)
        arr = rng.integers(0, 2, size=(10, 20), dtype=np.uint8)
        packed = pack_rows_to_ints(arr)
        unpacked = unpack_batch_of_ints_to_array(packed, 20)
        assert np.array_equal(unpacked, arr)

    def test_empty_batch(self):
        packed = []
        unpacked = unpack_batch_of_ints_to_array(packed, 8)
        assert unpacked.shape == (0, 8)

    def test_single_row(self):
        arr = np.array([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.uint8)
        packed = pack_rows_to_ints(arr)
        unpacked = unpack_batch_of_ints_to_array(packed, 8)
        assert np.array_equal(unpacked, arr)

    def test_wide_matrix(self):
        """Test with m > 64 bits to exercise the fallback path."""
        rng = np.random.default_rng(99)
        arr = rng.integers(0, 2, size=(5, 100), dtype=np.uint8)
        packed = pack_rows_to_ints(arr)
        unpacked = unpack_batch_of_ints_to_array(packed, 100)
        assert np.array_equal(unpacked, arr)


class TestUnpackFast:
    def test_matches_standard_small(self):
        """For m <= 64, fast version should match standard version."""
        rng = np.random.default_rng(7)
        arr = rng.integers(0, 2, size=(15, 32), dtype=np.uint8)
        packed = pack_rows_to_ints(arr)

        standard = unpack_batch_of_ints_to_array(packed, 32)
        fast = unpack_batch_of_ints_to_array_fast(packed, 32)
        assert np.array_equal(standard, fast)

    def test_matches_standard_64bit(self):
        rng = np.random.default_rng(11)
        arr = rng.integers(0, 2, size=(8, 64), dtype=np.uint8)
        packed = pack_rows_to_ints(arr)

        standard = unpack_batch_of_ints_to_array(packed, 64)
        fast = unpack_batch_of_ints_to_array_fast(packed, 64)
        assert np.array_equal(standard, fast)

    def test_falls_back_for_large_m(self):
        """For m > 64, fast should fall back to standard and still be correct."""
        rng = np.random.default_rng(13)
        arr = rng.integers(0, 2, size=(4, 80), dtype=np.uint8)
        packed = pack_rows_to_ints(arr)

        fast = unpack_batch_of_ints_to_array_fast(packed, 80)
        assert np.array_equal(fast, arr)

    def test_empty(self):
        result = unpack_batch_of_ints_to_array_fast([], 16)
        assert result.shape == (0, 16)
