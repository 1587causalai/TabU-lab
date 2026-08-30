"""Shared fail-closed checks for public evidence strings.

Public evidence may contain immutable remote URIs and repository-relative
pointers, but it must never disclose a host-local absolute path.  The checks
below deliberately recognize path *shapes* instead of enumerating familiar
private directories: an unusual mount such as ``/mnt`` or ``/srv`` is still a
private machine detail.
"""

from __future__ import annotations

import re

# ``\w`` is intentionally Unicode-aware.  An ASCII-only left boundary would
# treat ordinary prose such as ``输入/输出`` as if ``/输出`` started a path,
# while an ASCII-only segment would miss real paths such as ``/你好/秘密``.
_LEFT_TOKEN_BOUNDARY = r"(?<![\w._~+%/?#&:-])"
_LOCAL_FILE_URI = re.compile(
    rf"(?i){_LEFT_TOKEN_BOUNDARY}file:(?:/{{1,3}}|\\)"
)
_POSIX_ABSOLUTE = re.compile(
    rf"{_LEFT_TOKEN_BOUNDARY}/(?![/\s])"
)
_WINDOWS_ABSOLUTE = re.compile(
    rf"(?i){_LEFT_TOKEN_BOUNDARY}[a-z]:[\\/]"
)
_HOME_RELATIVE = re.compile(
    rf"{_LEFT_TOKEN_BOUNDARY}~[\\/]"
)
_UNC_ABSOLUTE = re.compile(
    # TeX line breaks are commonly written as ``\\\\[`` or ``\\\\{``
    # inside public mathematics fields.  Requiring a non-empty server segment
    # followed by a share separator handles Unicode hostnames while avoiding
    # ordinary TeX commands such as ``\\\\alpha``.
    rf"{_LEFT_TOKEN_BOUNDARY}(?:\\\\|//)(?=[^/\\\s{{}}\[\]()[\]^$]+[\\/])"
)
_REMOTE_URI = re.compile(
    r"(?i)(?<![\w.+-])(?!file:)[a-z][a-z0-9+.-]*://[^\s<>\"']+"
)
_SENSITIVE_KEY_FRAGMENTS = (
    "accesstoken",
    "apikey",
    "authtoken",
    "authorizationheader",
    "authorizationtoken",
    "bearertoken",
    "clientsecret",
    "cookie",
    "credential",
    "homedirectory",
    "hostname",
    "password",
    "passwd",
    "privatekey",
    "pythonexecutable",
    "secret",
    "sysexecutable",
    "username",
)
_SAFE_PUBLIC_KEYS = {
    # This is a typed, digest-only authorization *summary*, not an HTTP
    # Authorization header or credential.  Keep the exception exact so names
    # such as ``formal_authorization_token`` still fail closed.
    "formalauthorization",
}
_SENSITIVE_EXACT_KEYS = {"authorization"}

# Values need their own checks: a typed public field such as
# ``operating_system`` must not become a tunnel for a credential or an
# explicitly labelled private host identity.  Keep these recognizers narrow;
# ordinary hardware names and version strings are public evidence.
_PRIVATE_IDENTITY_VALUE = re.compile(
    r"(?i)(?<![a-z0-9])(?:host(?:name)?|user(?:name)?)\s*[:=]\s*\S+"
)
_URI_USERINFO = re.compile(
    r"(?i)(?<![\w.+-])[a-z][a-z0-9+.-]*://[^\s/@:]+(?::[^\s/@]*)?@"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?<![a-z0-9])(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"authorization|bearer[_-]?token|client[_-]?secret|cookie|credential|"
    r"passw(?:or)?d|private[_-]?key|secret)\s*[:=]\s*\S+"
)
_BEARER_CREDENTIAL = re.compile(r"(?i)(?<![a-z0-9])bearer\s+[a-z0-9._~+/-]{12,}")
_PEM_PRIVATE_KEY = re.compile(r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----")
_COMMON_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"AKIA[0-9A-Z]{16}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|sk-[A-Za-z0-9_-]{16,}"
    r")(?![A-Za-z0-9])"
)
_JWT_CREDENTIAL = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\."
    r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
)


def contains_local_file_uri(value: str) -> bool:
    """Return whether ``value`` contains a host-local ``file:`` URI."""

    return _LOCAL_FILE_URI.search(value) is not None


def contains_absolute_local_path(value: str) -> bool:
    """Return whether ``value`` contains any local absolute-path shape.

    Remote schemes such as ``https://``, ``hf://``, and ``git://`` are not
    paths: their double slash is preceded by a scheme colon and therefore does
    not match the UNC detector.  Repository-relative pointers likewise remain
    valid.
    """

    # A public URI can contain arbitrary Unicode path segments and slash-rich
    # schema identifiers.  Remove complete non-file URI tokens before looking
    # for local path shapes; a local path elsewhere in the same string remains
    # visible and is still rejected.
    without_remote_uris = _REMOTE_URI.sub("", value)
    return any(
        pattern.search(without_remote_uris) is not None
        for pattern in (
            _LOCAL_FILE_URI,
            _POSIX_ABSOLUTE,
            _WINDOWS_ABSOLUTE,
            _HOME_RELATIVE,
            _UNC_ABSOLUTE,
        )
    )


def is_sensitive_public_key(value: object) -> bool:
    """Return whether a mapping key has a secret or host-identity shape."""

    if not isinstance(value, str):
        return False
    normalized = re.sub(r"[^a-z0-9]+", "", value.casefold())
    if normalized in _SAFE_PUBLIC_KEYS:
        return False
    if normalized in _SENSITIVE_EXACT_KEYS:
        return True
    return any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS)


def contains_private_identity_or_secret(value: str) -> bool:
    """Return whether a string exposes an identity label or credential shape.

    This deliberately does not guess whether an arbitrary bare word is a
    person's name.  Typed environment fields constrain their own semantics;
    this shared detector covers explicit host/user labels, URI userinfo, and
    credential formats that are unsafe regardless of their field name.
    """

    return any(
        pattern.search(value) is not None
        for pattern in (
            _PRIVATE_IDENTITY_VALUE,
            _URI_USERINFO,
            _SECRET_ASSIGNMENT,
            _BEARER_CREDENTIAL,
            _PEM_PRIVATE_KEY,
            _COMMON_TOKEN,
            _JWT_CREDENTIAL,
        )
    )


__all__ = [
    "contains_absolute_local_path",
    "contains_local_file_uri",
    "contains_private_identity_or_secret",
    "is_sensitive_public_key",
]
