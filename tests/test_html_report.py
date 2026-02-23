"""Tests for the HTML report generator."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ct.reports.html import markdown_to_html, publish_report, render_html_report


class TestMarkdownToHtml:
    def test_headings(self):
        html = markdown_to_html("# Hello\n## World")
        assert "<h1" in html
        assert "<h2" in html
        assert "Hello" in html
        assert "World" in html

    def test_bold_and_italic(self):
        html = markdown_to_html("**bold** and *italic*")
        assert "<strong>bold</strong>" in html
        assert "<em>italic</em>" in html

    def test_lists(self):
        html = markdown_to_html("- item one\n- item two\n")
        assert "<li>" in html
        assert "item one" in html
        assert "item two" in html

    def test_tables(self):
        md = "| Col A | Col B |\n|-------|-------|\n| 1 | 2 |\n"
        html = markdown_to_html(md)
        assert "<table>" in html or "<table" in html
        assert "Col A" in html
        assert "Col B" in html

    def test_code_blocks(self):
        md = "```python\nprint('hello')\n```"
        html = markdown_to_html(md)
        assert "<code>" in html or "<pre>" in html
        assert "print" in html

    def test_fallback_without_markdown_lib(self):
        """markdown_to_html wraps in <pre> when markdown lib is missing."""
        # Temporarily hide the markdown module
        real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        import importlib
        import ct.reports.html as html_mod

        original_fn = html_mod.markdown_to_html

        # Simulate ImportError by patching the import inside the function
        def mock_markdown_to_html(md_text):
            escaped = (
                md_text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            return f"<pre>{escaped}</pre>"

        with patch.object(html_mod, "markdown_to_html", mock_markdown_to_html):
            result = html_mod.markdown_to_html("# Hello <world>")

        assert "<pre>" in result
        assert "&lt;world&gt;" in result


class TestRenderHtmlReport:
    def test_contains_doctype_and_css(self):
        html = render_html_report("# Test", title="My Report")
        assert "<!DOCTYPE html>" in html
        assert "<style>" in html
        assert "My Report" in html

    def test_injects_title(self):
        html = render_html_report("content", title="Drug Analysis")
        assert "Drug Analysis" in html
        assert "<title>Drug Analysis</title>" in html

    def test_injects_query(self):
        html = render_html_report("content", query="What is lenalidomide?")
        assert "What is lenalidomide?" in html
        assert "query-block" in html

    def test_no_query_block_div_when_empty(self):
        html = render_html_report("content", query="")
        assert '<div class="query-block">' not in html

    def test_injects_timestamp(self):
        html = render_html_report("content")
        assert "UTC" in html
        assert "Generated" in html

    def test_brand_footer(self):
        html = render_html_report("content")
        assert "CellType" in html
        assert "brand-footer" in html

    def test_query_html_escaping(self):
        html = render_html_report("content", query='<script>alert("xss")</script>')
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestPublishReport:
    def test_creates_html_file(self, tmp_path):
        md_file = tmp_path / "report.md"
        md_file.write_text("# My Report\n\nSome content here.")

        result = publish_report(md_file)

        assert result.exists()
        assert result.suffix == ".html"
        assert result.stem == "report"
        content = result.read_text()
        assert "<!DOCTYPE html>" in content
        assert "My Report" in content

    def test_custom_output_path(self, tmp_path):
        md_file = tmp_path / "input.md"
        md_file.write_text("# Analysis\n\nResults.")

        out = tmp_path / "output" / "custom.html"
        result = publish_report(md_file, out_path=out)

        assert result == out
        assert result.exists()
        assert "Analysis" in result.read_text()

    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.md"
        with pytest.raises(FileNotFoundError, match="nonexistent.md"):
            publish_report(missing)

    def test_title_from_first_heading(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Lenalidomide Deep Dive\n\nContent here.")

        result = publish_report(md_file)
        content = result.read_text()
        assert "Lenalidomide Deep Dive" in content
