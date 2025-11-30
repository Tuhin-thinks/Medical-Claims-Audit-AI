def create_file_hash(file_bytes: bytes) -> str:
    import hashlib

    sha256_hash = hashlib.sha256()
    sha256_hash.update(file_bytes)
    return sha256_hash.hexdigest()
