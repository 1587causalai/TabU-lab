from __future__ import annotations

import pytest

from tabu_lab.evidence.public_safety import (
    contains_absolute_local_path,
    contains_local_file_uri,
    contains_private_identity_or_secret,
)


@pytest.mark.parametrize(
    "value",
    (
        "/你好/秘密",
        "/équipe/secret",
        "/💾/secret",
        "checkpoint: /研究/模型.safetensors",
        r"C:\Users\张三\秘密.txt",
        "D:/équipe/secret.txt",
        r"\\服务器\共享\秘密.txt",
        "//服务器/共享/秘密.txt",
        "~/私密/checkpoint.bin",
        "file:///Users/张三/秘密.txt",
        r"file:\\服务器\共享\秘密.txt",
    ),
)
def test_absolute_local_path_detection_is_unicode_and_platform_complete(value: str) -> None:
    assert contains_absolute_local_path(value)


@pytest.mark.parametrize(
    "value",
    (
        "https://example.org/équipe/公开",
        "https://[2001:db8::1]/你好/schema.json",
        "hf://datasets/wehub/公开-data@revision/snapshot.json",
        "git://example.org/研究/tabu-lab.git",
        "$schema=https://json-schema.org/draft/2020-12/schema",
        "tabu://schema/eval-result/v1",
        "tabu.eval-result/v1",
        "输入/输出 and models/runs",
        "classification/regression/completion",
        "models / experiments / runs",
        "RMSE / MAE",
        "use / for division",
    ),
)
def test_absolute_local_path_detection_preserves_public_uris_and_slash_prose(
    value: str,
) -> None:
    assert not contains_absolute_local_path(value)


def test_remote_uri_does_not_hide_separate_unicode_local_path() -> None:
    value = "source https://example.org/公开/schema.json; cache /你好/秘密"
    assert contains_absolute_local_path(value)


@pytest.mark.parametrize(
    "value",
    (
        "file:///Users/张三/秘密.txt",
        r"file:\\服务器\共享\秘密.txt",
        "artifact=file:/équipe/secret.txt",
    ),
)
def test_local_file_uri_detection_covers_unicode_and_windows(value: str) -> None:
    assert contains_local_file_uri(value)


@pytest.mark.parametrize(
    "value",
    (
        "https://example.org/file/公开",
        "https://example.org/schema/file:v1",
        "profile/field:file/value",
    ),
)
def test_local_file_uri_detection_preserves_remote_and_relative_identifiers(
    value: str,
) -> None:
    assert not contains_local_file_uri(value)


@pytest.mark.parametrize(
    "value",
    (
        "hostname=gongqian-mini",
        "username: cms",
        "ssh://alice@example.org/research.git",
        "password=hunter2",
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        "Bearer abcdefghijklmnopqrstuvwxyz",
        "sk-abcdefghijklmnopqrstuvwxyz123456",
        "ghp_abcdefghijklmnopqrstuvwxyz123456",
        "AKIAIOSFODNN7EXAMPLE",
        "-----BEGIN PRIVATE KEY-----",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature123",
    ),
)
def test_private_identity_and_secret_detection_rejects_public_string_leaks(
    value: str,
) -> None:
    assert contains_private_identity_or_secret(value)


@pytest.mark.parametrize(
    "value",
    (
        "Darwin",
        "cpu-host",
        "arm64",
        "cuda:0",
        "Apple Metal Performance Shaders",
        "NVIDIA GeForce RTX 4090",
        "adult-v2-task-7592-classification-micro-base",
        "3.11.14",
        "2.8.0+cu128",
        "12.8",
    ),
)
def test_private_identity_and_secret_detection_preserves_public_environment_values(
    value: str,
) -> None:
    assert not contains_private_identity_or_secret(value)
