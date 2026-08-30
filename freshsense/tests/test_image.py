"""Untrusted-input handling at the image boundary."""

from __future__ import annotations

import pytest

from freshsense.vision.predict import MAX_IMAGE_BYTES, InvalidImageError, load_image

from .conftest import make_image


def test_a_valid_jpeg_loads_as_rgb():
    image = load_image(make_image())
    assert image.mode == "RGB"
    assert image.size == (64, 64)


def test_empty_upload_is_rejected():
    with pytest.raises(InvalidImageError, match="empty upload"):
        load_image(b"")


def test_non_image_bytes_are_rejected():
    with pytest.raises(InvalidImageError, match="could not decode"):
        load_image(b"#!/bin/sh\nrm -rf /\n")


def test_truncated_image_is_rejected():
    with pytest.raises(InvalidImageError):
        load_image(make_image()[:40])


def test_oversized_upload_is_rejected_before_decoding():
    with pytest.raises(InvalidImageError, match="MB limit"):
        load_image(b"\xff" * (MAX_IMAGE_BYTES + 1))
