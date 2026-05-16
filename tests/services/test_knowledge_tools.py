"""Tests for app.services.knowledge_tools — Gap 4 (file size guard) and Gap 10 (stat guard)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ── list_domain_documents (Gap 10: per-file stat guard) ──────────────── #


class TestListDomainDocuments:
    def test_returns_empty_list_when_dir_does_not_exist(self):
        from app.services.knowledge_tools import list_domain_documents

        with patch("app.services.knowledge_tools.settings") as mock_settings:
            mock_settings.domain_docs_dir = Path("/nonexistent/path/xyz")
            result = json.loads(list_domain_documents())
        assert result == []

    def test_returns_filename_and_size_for_valid_file(self, tmp_path):
        from app.services.knowledge_tools import list_domain_documents

        doc = tmp_path / "guide.md"
        doc.write_text("# Hello")

        with patch("app.services.knowledge_tools.settings") as mock_settings:
            mock_settings.domain_docs_dir = tmp_path
            result = json.loads(list_domain_documents())

        assert len(result) == 1
        assert result[0]["file_name"] == "guide.md"
        assert result[0]["size_bytes"] == len("# Hello")

    def test_returns_zero_size_when_stat_raises_os_error(self, tmp_path):
        """Gap 10: per-file stat error in the size-check block yields size=0, not crash."""
        from app.services.knowledge_tools import list_domain_documents

        doc = tmp_path / "broken.md"
        doc.write_text("content")

        # Patch Path.iterdir to yield a fake path whose stat() raises OSError,
        # but whose is_file() returns True so it passes the filter.
        fake_path = MagicMock(spec=Path)
        fake_path.is_file.return_value = True
        fake_path.suffix = ".md"
        fake_path.name = "broken.md"
        fake_path.stat.side_effect = OSError("Permission denied")
        # Make it sortable
        fake_path.__lt__ = lambda self, other: True

        with patch("app.services.knowledge_tools.settings") as mock_settings:
            mock_settings.domain_docs_dir = tmp_path
            with patch.object(Path, "iterdir", return_value=[fake_path]):
                result = json.loads(list_domain_documents())

        assert len(result) == 1
        assert result[0]["size_bytes"] == 0

    def test_ignores_non_text_files(self, tmp_path):
        from app.services.knowledge_tools import list_domain_documents

        (tmp_path / "data.csv").write_text("a,b")
        (tmp_path / "doc.md").write_text("# Doc")

        with patch("app.services.knowledge_tools.settings") as mock_settings:
            mock_settings.domain_docs_dir = tmp_path
            result = json.loads(list_domain_documents())

        names = [r["file_name"] for r in result]
        assert "doc.md" in names
        assert "data.csv" not in names


# ── list_datasets (Gap 10: per-file stat guard) ───────────────────────── #


class TestListDatasets:
    def test_returns_empty_list_when_dir_does_not_exist(self):
        from app.services.knowledge_tools import list_datasets

        with patch("app.services.knowledge_tools.settings") as mock_settings:
            mock_settings.datasets_dir = Path("/nonexistent/xyz")
            result = json.loads(list_datasets())

        assert result == []

    def test_returns_zero_size_when_stat_raises(self, tmp_path):
        """Gap 10: stat failure on dataset file must yield size=0."""
        from app.services.knowledge_tools import list_datasets

        (tmp_path / "data.csv").write_text("col1,col2\n1,2")

        fake_path = MagicMock(spec=Path)
        fake_path.is_file.return_value = True
        fake_path.suffix = ".csv"
        fake_path.name = "data.csv"
        fake_path.stat.side_effect = OSError("No access")
        fake_path.__lt__ = lambda self, other: True

        with patch("app.services.knowledge_tools.settings") as mock_settings:
            mock_settings.datasets_dir = tmp_path
            with patch.object(Path, "iterdir", return_value=[fake_path]):
                result = json.loads(list_datasets())

        assert result[0]["size_bytes"] == 0

    def test_returns_correct_format_for_csv(self, tmp_path):
        from app.services.knowledge_tools import list_datasets

        (tmp_path / "sample.csv").write_text("a,b\n1,2")

        with patch("app.services.knowledge_tools.settings") as mock_settings:
            mock_settings.datasets_dir = tmp_path
            result = json.loads(list_datasets())

        assert result[0]["format"] == "csv"


# ── inspect_dataset (Gap 4: file size guard) ─────────────────────────── #


class TestInspectDataset:
    def test_rejects_file_exceeding_size_limit(self, tmp_path):
        """Gap 4: files larger than max_dataset_bytes must return an error string."""
        import pandas as pd
        from app.services.knowledge_tools import inspect_dataset

        csv_path = tmp_path / "big.csv"
        csv_path.write_text("col\n" + "\n".join(str(i) for i in range(100)))

        with patch("app.services.knowledge_tools.settings") as mock_settings:
            mock_settings.datasets_dir = tmp_path
            mock_settings.max_dataset_bytes = 1  # 1-byte limit forces rejection
            result = inspect_dataset("big.csv")

        assert result.startswith("Error:")
        assert "too large" in result

    def test_accepts_file_within_size_limit(self, tmp_path):
        """File smaller than limit must be loaded and return valid JSON."""
        from app.services.knowledge_tools import inspect_dataset

        csv_path = tmp_path / "small.csv"
        csv_path.write_text("value\n1\n2\n3")

        with patch("app.services.knowledge_tools.settings") as mock_settings:
            mock_settings.datasets_dir = tmp_path
            mock_settings.max_dataset_bytes = 10 * 1024 * 1024  # 10 MB
            data = json.loads(inspect_dataset("small.csv"))

        assert data["rows"] == 3
        assert data["columns"] == 1

    def test_returns_error_for_missing_file(self, tmp_path):
        from app.services.knowledge_tools import inspect_dataset

        with patch("app.services.knowledge_tools.settings") as mock_settings:
            mock_settings.datasets_dir = tmp_path
            mock_settings.max_dataset_bytes = 10 * 1024 * 1024
            result = inspect_dataset("nonexistent.csv")

        assert "Error:" in result
        assert "not found" in result

    def test_returns_error_for_path_traversal_attempt(self, tmp_path):
        from app.services.knowledge_tools import inspect_dataset

        with patch("app.services.knowledge_tools.settings") as mock_settings:
            mock_settings.datasets_dir = tmp_path
            mock_settings.max_dataset_bytes = 10 * 1024 * 1024
            result = inspect_dataset("../../etc/passwd")

        assert "Error:" in result
        assert "invalid path" in result.lower() or "path" in result.lower()

    def test_returns_error_when_stat_raises_os_error(self, tmp_path):
        """stat() failure in the size-check block must return an Error: string."""
        from app.services.knowledge_tools import inspect_dataset

        csv_path = tmp_path / "unreadable.csv"
        csv_path.write_text("a\n1")

        # Make exists() bypass stat() by returning True directly,
        # then make the explicit stat() call raise OSError.
        with patch("app.services.knowledge_tools.settings") as mock_settings:
            mock_settings.datasets_dir = tmp_path
            mock_settings.max_dataset_bytes = 10 * 1024 * 1024

            with patch.object(Path, "exists", return_value=True):
                original_stat = Path.stat

                def stat_raises(self, **kwargs):
                    if self.name == "unreadable.csv":
                        raise OSError("Permission denied")
                    return original_stat(self, **kwargs)

                with patch.object(Path, "stat", stat_raises):
                    result = inspect_dataset("unreadable.csv")

        assert result.startswith("Error:")


# ── describe_columns (Gap 4: file size guard) ────────────────────────── #


class TestDescribeColumns:
    def test_rejects_oversized_file(self, tmp_path):
        """Gap 4: describe_columns must also enforce the size limit."""
        from app.services.knowledge_tools import describe_columns

        csv_path = tmp_path / "huge.csv"
        csv_path.write_text("val\n" + "\n".join(str(i) for i in range(50)))

        with patch("app.services.knowledge_tools.settings") as mock_settings:
            mock_settings.datasets_dir = tmp_path
            mock_settings.max_dataset_bytes = 1  # 1-byte limit
            result = describe_columns("huge.csv", ["val"])

        assert "Error:" in result
        assert "too large" in result

    def test_accepts_small_file_and_returns_stats(self, tmp_path):
        from app.services.knowledge_tools import describe_columns

        csv_path = tmp_path / "nums.csv"
        csv_path.write_text("score\n10\n20\n30")

        with patch("app.services.knowledge_tools.settings") as mock_settings:
            mock_settings.datasets_dir = tmp_path
            mock_settings.max_dataset_bytes = 10 * 1024 * 1024
            result = json.loads(describe_columns("nums.csv", ["score"]))

        assert "score" in result
        assert result["score"]["type"] == "numeric"
        assert result["score"]["mean"] == pytest.approx(20.0)
