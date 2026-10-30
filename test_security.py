import time
import unittest
from unittest import mock

from app.security import create_token, decode_token, hash_password, verify_password


class SecurityTests(unittest.TestCase):
    def test_password_roundtrip(self):
        stored = hash_password("secret123")
        self.assertTrue(verify_password("secret123", stored))
        self.assertFalse(verify_password("wrong", stored))

    def test_token_roundtrip_and_tamper(self):
        token = create_token({"sub": 7}, "k", 1)
        self.assertEqual(decode_token(token, "k")["sub"], 7)
        self.assertIsNone(decode_token(token, "other-key"))
        head, body, sig = token.split(".")
        self.assertIsNone(decode_token(f"{head}.{body}x.{sig}", "k"))

    def test_token_expiry(self):
        token = create_token({"sub": 1}, "k", 1)
        with mock.patch("time.time", return_value=time.time() + 2 * 3600):
            self.assertIsNone(decode_token(token, "k"))


if __name__ == "__main__":
    unittest.main()
