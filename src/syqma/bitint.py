"""Pack and unpack binary arrays to Python integer representations."""

from collections.abc import Sequence

import numpy as np


def unpack_batch_of_ints_to_array_fast(
    ops_as_ints: Sequence[int] | np.ndarray,
    m: int,
) -> np.ndarray:
    """Unpack integer-encoded rows into a binary array.

    Uses vectorized bit extraction for rows up to 64 bits and falls back to the
    byte-based implementation for wider rows.
    """
    batch_size = len(ops_as_ints)
    if batch_size == 0:
        return np.empty((0, m), dtype=np.uint8)

    if m <= 64:
        # NumPy broadcasts each integer against all bit positions at once.
        arr = np.asarray(ops_as_ints, dtype=np.uint64)
        bit_positions = np.arange(m, dtype=np.uint64)
        bits = ((arr[:, None] >> bit_positions) & 1).astype(np.uint8)
        return bits

    # Fall back to original method for larger m
    return unpack_batch_of_ints_to_array(ops_as_ints, m)


def pack_rows_to_ints(arr: np.ndarray) -> list[int]:
    """Pack each binary row into a little-endian Python integer."""
    pb = np.packbits(arr, axis=1, bitorder="little")  # shape (d, B) where B = ceil(m/8)
    # int.from_bytes does the final conversion in C for each row.
    packed = [
        int.from_bytes(memoryview(pb[i]), byteorder="little")
        for i in range(pb.shape[0])
    ]

    return packed


def pack_array_to_int(x: np.ndarray) -> int:
    """Pack a binary vector into a little-endian Python integer."""
    b = np.packbits(x, bitorder="little")

    return int.from_bytes(memoryview(b), "little")


def unpack_int_to_array(x_int: int, m: int) -> np.ndarray:
    """Unpack a little-endian integer into a length-``m`` binary vector."""
    out = np.empty(m, dtype=np.uint8)
    v = int(x_int)
    for i in range(m):
        out[i] = v & 1
        v >>= 1

    return out


def unpack_batch_of_ints_to_array(
    ops_as_ints: Sequence[int] | np.ndarray,
    m: int,
) -> np.ndarray:
    """Unpack integer-encoded rows into a two-dimensional binary array."""
    batch_size = len(ops_as_ints)
    if batch_size == 0:
        return np.empty((0, m), dtype=np.uint8)

    # Calculate the number of bytes needed for m bits.
    n_bytes = (m + 7) // 8

    buf = bytearray(batch_size * n_bytes)
    mv = memoryview(buf)
    for i, op in enumerate(ops_as_ints):
        mv[i * n_bytes : (i + 1) * n_bytes] = int(op).to_bytes(n_bytes, "little")

    # Use frombuffer to create a 1D uint8 array from the byte string...
    byte_array = np.frombuffer(buf, dtype=np.uint8)
    # ...and reshape it to (batch_size, n_bytes).
    byte_array = byte_array.reshape(batch_size, n_bytes)

    # Unpack the entire 2D byte array into a 2D bit array in one shot.
    unpacked = np.unpackbits(byte_array, axis=1, bitorder="little")

    # Truncate to the correct number of bits and return.
    return unpacked[:, :m]
