"""encryption.py

Encryption utility for the Koios API.
Uses AES-256-GCM for authenticated encryption.

Author: Jared Paubel jpaubel@pm.me
"""
import base64
import json
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from src.config import config, logger

class Encryption:
    """Utility class for encrypting and decrypting data."""

    @staticmethod
    def _get_aes_gcm() -> AESGCM:
        """Initialize AESGCM with the configured encryption key."""
        key_hex = config.encryption_key
        if not key_hex:
            logger.error("ENCRYPTION_KEY is not set.")
            raise ValueError("ENCRYPTION_KEY not set in environment.")
        
        try:
            key = bytes.fromhex(key_hex)
        except ValueError:
            logger.error("ENCRYPTION_KEY is not a valid hex string.")
            raise ValueError("ENCRYPTION_KEY must be a valid hex string.")
        
        if len(key) != 32:
            logger.error("ENCRYPTION_KEY must be 32 bytes (64 hex characters).")
            raise ValueError("ENCRYPTION_KEY must be 32 bytes.")
        
        return AESGCM(key)

    @classmethod
    def encrypt(cls, data: dict) -> str:
        """Encrypt a dictionary into a base64-encoded string.

        The output format is: base64(nonce + ciphertext + tag)
        """
        aesgcm = cls._get_aes_gcm()
        nonce = os.urandom(12)  # GCM recommended nonce size
        data_json = json.dumps(data).encode('utf-8')
        
        # AESGCM.encrypt returns ciphertext + tag
        ciphertext_with_tag = aesgcm.encrypt(nonce, data_json, None)
        
        # Combine nonce and ciphertext+tag for storage/transmission
        combined = nonce + ciphertext_with_tag
        return base64.b64encode(combined).decode('utf-8')

    @classmethod
    def decrypt(cls, encrypted_str: str) -> dict:
        """Decrypt a base64-encoded string into a dictionary."""
        aesgcm = cls._get_aes_gcm()
        try:
            combined = base64.b64decode(encrypted_str)
            if len(combined) < 12:
                raise ValueError("Invalid encrypted data: too short.")
            
            nonce = combined[:12]
            ciphertext_with_tag = combined[12:]
            
            decrypted_data = aesgcm.decrypt(nonce, ciphertext_with_tag, None)
            return json.loads(decrypted_data.decode('utf-8'))
        except Exception as e:
            logger.error("Decryption failed: %s", e)
            raise ValueError("Decryption failed. Invalid data or key.")
