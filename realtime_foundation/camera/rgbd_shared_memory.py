from __future__ import annotations

import fcntl
import mmap
import os
from pathlib import Path
import re
import struct

import numpy as np


_MAGIC = b"FPRGBD1\0"
_VERSION = 1
_HEADER_SIZE = 128
_HEADER = struct.Struct("<8sIIIIIQd9f")
_NO_ACTIVE_SLOT = 0xFFFFFFFF
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class LatestRgbdFrameBuffer:
  def __init__(self, name: str, width: int, height: int, slot_count: int, create: bool):
    if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
      raise ValueError(f"Invalid RGB-D shared-memory name: {name!r}")
    self.name = name
    self.width = int(width)
    self.height = int(height)
    if self.width <= 0 or self.height <= 0:
      raise ValueError("Shared RGB-D width and height must be positive")
    self.slot_count = int(slot_count)
    if self.slot_count < 2:
      raise ValueError("Shared RGB-D slot_count must be at least 2")

    self.path = Path("/dev/shm") / self.name
    self.rgb_shape = (self.height, self.width, 3)
    self.depth_shape = (self.height, self.width)
    self.rgb_nbytes = int(np.prod(self.rgb_shape, dtype=np.int64))
    self.depth_nbytes = int(np.prod(self.depth_shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    self.slot_nbytes = self.rgb_nbytes + self.depth_nbytes
    self.total_size = _HEADER_SIZE + self.slot_count * self.slot_nbytes
    self._owner = bool(create)
    self._closed = False

    flags = os.O_RDWR
    if create:
      flags |= os.O_CREAT | os.O_EXCL
    self.fd = os.open(self.path, flags, 0o600)
    try:
      if create:
        os.ftruncate(self.fd, self.total_size)
      else:
        actual_size = os.fstat(self.fd).st_size
        if actual_size != self.total_size:
          raise RuntimeError(
              f"RGB-D shared-memory size mismatch: expected {self.total_size}, got {actual_size}"
          )
      self.mapping = mmap.mmap(self.fd, self.total_size, access=mmap.ACCESS_WRITE)
      if create:
        self._write_header(
            active_slot=_NO_ACTIVE_SLOT,
            sequence=0,
            timestamp=0.0,
            K=np.eye(3, dtype=np.float32),
        )
      else:
        self._validate_header()
    except Exception:
      os.close(self.fd)
      if create:
        self.path.unlink(missing_ok=True)
      raise

  @classmethod
  def create(
      cls,
      name: str,
      width: int,
      height: int,
      slot_count: int,
  ) -> "LatestRgbdFrameBuffer":
    return cls(name=name, width=width, height=height, slot_count=slot_count, create=True)

  @classmethod
  def attach(
      cls,
      name: str,
      width: int,
      height: int,
      slot_count: int,
  ) -> "LatestRgbdFrameBuffer":
    return cls(name=name, width=width, height=height, slot_count=slot_count, create=False)

  def write(self, color: np.ndarray, depth: np.ndarray, K: np.ndarray, timestamp: float) -> int:
    color_array = np.asarray(color)
    depth_array = np.asarray(depth)
    K_array = np.asarray(K, dtype=np.float32)
    if color_array.shape != self.rgb_shape or color_array.dtype != np.uint8:
      raise ValueError(f"Expected RGB {self.rgb_shape}/uint8, got {color_array.shape}/{color_array.dtype}")
    if depth_array.shape != self.depth_shape or depth_array.dtype != np.float32:
      raise ValueError(
          f"Expected depth {self.depth_shape}/float32, got {depth_array.shape}/{depth_array.dtype}"
      )
    if K_array.shape != (3, 3):
      raise ValueError(f"Expected K shape (3, 3), got {K_array.shape}")

    self._lock_metadata(fcntl.LOCK_EX)
    try:
      active_slot, sequence, _, _ = self._read_header_unlocked()
      target_slot = 0 if active_slot == _NO_ACTIVE_SLOT else (active_slot + 1) % self.slot_count
    finally:
      self._unlock_metadata()

    self._lock_slot(target_slot, fcntl.LOCK_EX)
    try:
      rgb_offset, depth_offset = self._slot_offsets(target_slot)
      depth_view = np.ndarray(
          self.depth_shape,
          dtype=np.float32,
          buffer=self.mapping,
          offset=depth_offset,
      )
      rgb_view = np.ndarray(self.rgb_shape, dtype=np.uint8, buffer=self.mapping, offset=rgb_offset)
      np.copyto(rgb_view, color_array, casting="no")
      np.copyto(depth_view, depth_array, casting="no")
      sequence += 1
      self._lock_metadata(fcntl.LOCK_EX)
      try:
        self._write_header(
            active_slot=target_slot,
            sequence=sequence,
            timestamp=float(timestamp),
            K=K_array,
        )
      finally:
        self._unlock_metadata()
      return sequence
    finally:
      self._unlock_slot(target_slot)

  def read_latest(
      self,
      last_sequence: int,
  ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float] | None:
    self._lock_metadata(fcntl.LOCK_SH)
    try:
      active_slot, sequence, timestamp, K = self._read_header_unlocked()
      if active_slot == _NO_ACTIVE_SLOT or sequence == int(last_sequence):
        return None
      self._lock_slot(active_slot, fcntl.LOCK_SH)
    finally:
      self._unlock_metadata()

    try:
      rgb_offset, depth_offset = self._slot_offsets(active_slot)
      color = np.ndarray(
          self.rgb_shape,
          dtype=np.uint8,
          buffer=self.mapping,
          offset=rgb_offset,
      ).copy()
      depth = np.ndarray(
          self.depth_shape,
          dtype=np.float32,
          buffer=self.mapping,
          offset=depth_offset,
      ).copy()
      return sequence, color, depth, K, timestamp
    finally:
      self._unlock_slot(active_slot)

  def close(self, unlink: bool = False) -> None:
    if self._closed:
      return
    first_error = None
    try:
      self.mapping.close()
    except Exception as exc:
      first_error = exc
    try:
      os.close(self.fd)
    except OSError as exc:
      if first_error is None:
        first_error = exc
    if unlink:
      try:
        self.path.unlink(missing_ok=True)
      except OSError as exc:
        if first_error is None:
          first_error = exc
    self._closed = True
    if first_error is not None:
      raise first_error

  def _write_header(
      self,
      active_slot: int,
      sequence: int,
      timestamp: float,
      K: np.ndarray,
  ) -> None:
    K_values = np.asarray(K, dtype=np.float32).reshape(-1).tolist()
    _HEADER.pack_into(
        self.mapping,
        0,
        _MAGIC,
        _VERSION,
        self.width,
        self.height,
        self.slot_count,
        int(active_slot),
        int(sequence),
        float(timestamp),
        *K_values,
    )

  def _read_header_unlocked(self) -> tuple[int, int, float, np.ndarray]:
    values = _HEADER.unpack_from(self.mapping, 0)
    magic, version, width, height, slot_count, active_slot, sequence, timestamp, *K_values = values
    if magic != _MAGIC or version != _VERSION:
      raise RuntimeError(
          f"Invalid RGB-D shared-memory header: magic={magic!r}, version={version}"
      )
    if width != self.width or height != self.height:
      raise RuntimeError(
          f"RGB-D shared-memory shape mismatch: expected {self.width}x{self.height}, "
          f"got {width}x{height}"
      )
    if slot_count != self.slot_count:
      raise RuntimeError(
          f"RGB-D shared-memory slot mismatch: expected {self.slot_count}, got {slot_count}"
      )
    if active_slot != _NO_ACTIVE_SLOT and active_slot >= self.slot_count:
      raise RuntimeError(f"Invalid RGB-D shared-memory active slot: {active_slot}")
    K = np.asarray(K_values, dtype=np.float32).reshape(3, 3)
    return int(active_slot), int(sequence), float(timestamp), K

  def _validate_header(self) -> None:
    self._lock_metadata(fcntl.LOCK_SH)
    try:
      self._read_header_unlocked()
    finally:
      self._unlock_metadata()

  def _slot_offsets(self, slot_index: int) -> tuple[int, int]:
    rgb_offset = _HEADER_SIZE + int(slot_index) * self.slot_nbytes
    return rgb_offset, rgb_offset + self.rgb_nbytes

  def _lock_metadata(self, operation: int) -> None:
    fcntl.lockf(self.fd, operation, 1, 0, os.SEEK_SET)

  def _unlock_metadata(self) -> None:
    fcntl.lockf(self.fd, fcntl.LOCK_UN, 1, 0, os.SEEK_SET)

  def _lock_slot(self, slot_index: int, operation: int) -> None:
    fcntl.lockf(self.fd, operation, 1, 1 + int(slot_index), os.SEEK_SET)

  def _unlock_slot(self, slot_index: int) -> None:
    fcntl.lockf(self.fd, fcntl.LOCK_UN, 1, 1 + int(slot_index), os.SEEK_SET)
