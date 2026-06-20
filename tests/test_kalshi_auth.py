"""Tests for Kalshi RSA-PSS request signing (ingestion/kalshi.py).

A fresh RSA key is generated in-test, used to sign a message, and the signature is
verified with the matching public key — proving the signing mechanics are correct
without any real credentials or network access.
"""

from __future__ import annotations

import base64

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from ingestion import kalshi


@pytest.fixture
def rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _verify(public_key: rsa.RSAPublicKey, message: str, signature_b64: str) -> None:
    public_key.verify(
        base64.b64decode(signature_b64),
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )


def test_signature_verifies_with_public_key(rsa_key: rsa.RSAPrivateKey) -> None:
    message = "1718000000000GET/trade-api/v2/portfolio/balance"
    signature = kalshi.sign_message(rsa_key, message)
    # Should not raise.
    _verify(rsa_key.public_key(), message, signature)


def test_tampered_message_fails_verification(rsa_key: rsa.RSAPrivateKey) -> None:
    signature = kalshi.sign_message(rsa_key, "original-message")
    with pytest.raises(InvalidSignature):
        _verify(rsa_key.public_key(), "tampered-message", signature)


def test_signed_headers_structure(rsa_key: rsa.RSAPrivateKey) -> None:
    headers = kalshi.signed_headers(
        "GET",
        "/trade-api/v2/portfolio/balance",
        private_key=rsa_key,
        key_id="test-key-id",
        timestamp_ms=1718000000000,
    )
    assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == "1718000000000"
    # The signature must verify against the canonical message string.
    message = "1718000000000GET/trade-api/v2/portfolio/balance"
    _verify(rsa_key.public_key(), message, headers["KALSHI-ACCESS-SIGNATURE"])


def test_signed_headers_requires_credentials() -> None:
    with pytest.raises(RuntimeError):
        kalshi.signed_headers(
            "GET", "/trade-api/v2/portfolio/balance", private_key=None, key_id=""
        )


def test_load_private_key_from_pem(rsa_key: rsa.RSAPrivateKey) -> None:
    from cryptography.hazmat.primitives import serialization

    pem = rsa_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    loaded = kalshi.load_private_key(pem)
    assert isinstance(loaded, rsa.RSAPrivateKey)


def test_load_private_key_missing_returns_none() -> None:
    assert kalshi.load_private_key("") is None
    assert kalshi.load_private_key("/no/such/key.pem") is None
