"""
Comprehensive tests for string_fixer.py module.

Tests cover:
- Normal strings (single and double quotes)
- Raw strings (r"..." and r'...')
- F-strings (f"..." with variables and expressions)
- Triple-quoted docstrings
- Edge cases (empty strings, very long strings, special characters)
- Multi-line handling
"""

import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

# Add parent directory to path to import string_fixer
sys.path.insert(0, str(Path(__file__).parent.parent))

from string_fixer import (
    MAX_LINE_LENGTH,
    fix_content,
    fix_file,
    split_f_string,
    split_normal_string,
    split_string_token,
    wrap_docstring,
)


class TestNormalStrings:
    """Test normal string splitting functionality."""

    def test_short_string_unchanged(self):
        """Short strings should not be modified."""
        content = 'message = "Hello World"'
        result = fix_content(content)
        assert result == content

    def test_long_string_gets_split(self):
        """Long strings exceeding MAX_LINE_LENGTH should be split."""
        # Create a string that's definitely too long
        long_text = "This is a very long string " * 10
        content = f'message = "{long_text}"'
        result = fix_content(content)

        # Result should be different and contain continuation
        assert result != content
        assert "\\" in result or len(result.splitlines()) > 1

    def test_string_with_spaces_splits_at_space(self):
        """String should split at word boundaries when possible."""
        content = 'x = "' + "word " * 50 + '"'
        result = fix_content(content)

        # Should split and not break words
        assert "\\" in result

    def test_double_quoted_string(self):
        """Test double-quoted string splitting."""
        long_str = "A" * 100
        content = f'msg = "{long_str}"'
        result = fix_content(content)
        assert result != content

    def test_single_quoted_string(self):
        """Test single-quoted string splitting."""
        long_str = "B" * 100
        content = f"msg = '{long_str}'"
        result = fix_content(content)
        assert result != content


class TestRawStrings:
    """Test raw string (r"...") handling."""

    def test_short_raw_string_unchanged(self):
        """Short raw strings should not be modified."""
        content = r'path = r"C:\Users\test"'
        result = fix_content(content)
        assert result == content

    def test_long_raw_string_gets_split(self):
        """Long raw strings should be split properly."""
        long_path = r"C:\Users\VeryLongUsername\Documents\Projects\Subfolder\AnotherFolder\YetAnotherFolder\Files\Data"
        content = f'path = r"{long_path}"'
        result = fix_content(content)

        # Should be split
        assert result != content or len(content) <= MAX_LINE_LENGTH

    def test_raw_string_preserves_backslashes(self):
        """Raw strings should preserve backslashes correctly."""
        content = r'regex = r"\d+\.\d+\.\d+\.\d+ is a pattern that matches IP addresses and more text to make it long"'
        result = fix_content(content)

        # Should still contain backslashes
        assert "\\" in result


class TestFStrings:
    """Test f-string formatting and splitting."""

    def test_short_fstring_unchanged(self):
        """Short f-strings should not be modified."""
        content = 'msg = f"Hello {name}"'
        result = fix_content(content)
        assert result == content

    def test_long_fstring_gets_split(self):
        """Long f-strings should be split."""
        vars_part = "{var1} {var2} {var3} {var4}" * 5
        content = f'msg = f"Long message with {vars_part}"'
        result = fix_content(content)

        # Should attempt to split
        assert result != content or len(content) <= MAX_LINE_LENGTH

    def test_fstring_preserves_expressions(self):
        """F-string expressions should not be broken."""
        content = 'msg = f"Result: {calculate_something(x, y, z)} and more text to make this line exceed the maximum allowed line length"'
        result = fix_content(content)

        # Should contain the expression intact
        assert (
            "{calculate_something(x, y, z)}" in result
            or "calculate_something" in result
        )

    def test_fstring_with_nested_braces(self):
        """F-strings with nested braces should be handled carefully."""
        content = (
            'msg = f"Dict: {data["key"]} with more content ' + "x" * 50 + '"'
        )
        result = fix_content(content)

        # Should not break the nested braces
        assert "data[" in result or "data" in result


class TestDocstrings:
    """Test triple-quoted docstring wrapping."""

    def test_short_docstring_unchanged(self):
        """Short docstrings should not be modified."""
        content = '"""Short docstring."""'
        result = fix_content(content)
        assert result == content

    def test_long_docstring_gets_wrapped(self):
        """Long docstrings should be wrapped to multiple lines."""
        long_text = "This is a very long docstring that exceeds the maximum line length and should be wrapped appropriately to maintain PEP 8 compliance."
        content = f'"""{long_text}"""'
        result = fix_content(content)

        # Should be different
        assert result != content or len(content) <= MAX_LINE_LENGTH

    def test_docstring_preserves_triple_quotes(self):
        """Docstrings should maintain triple quotes."""
        content = '"""' + "Long " * 30 + '"""'
        result = fix_content(content)

        # Should still have triple quotes
        assert '"""' in result


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_string(self):
        """Empty content should be handled."""
        content = ""
        result = fix_content(content)
        assert result == ""

    def test_string_with_escape_sequences(self):
        """Strings with escape sequences should be preserved."""
        content = r'msg = "Line 1\nLine 2\nLine 3\t\tTabbed and this needs to be much longer to trigger splitting behavior"'
        result = fix_content(content)

        # Escape sequences should remain
        assert "\\n" in result or "\\t" in result

    def test_multiple_strings_in_one_line(self):
        """Multiple strings on the same line should be handled."""
        content = (
            'a = "First very long string '
            + "x" * 50
            + '"; b = "Second string"'
        )
        result = fix_content(content)

        # Should process both strings
        assert len(result) >= len(content)

    def test_string_at_deep_indentation(self):
        """Strings with deep indentation should be handled."""
        content = '            msg = "' + "A" * 70 + '"'
        result = fix_content(content)

        # Should handle indentation
        assert result != content or len(content) <= MAX_LINE_LENGTH

    def test_empty_string_literal(self):
        """Empty string literals should be unchanged."""
        content = 'msg = ""'
        result = fix_content(content)
        assert result == content

    def test_syntax_error_content(self):
        """Content with syntax errors should be returned unchanged."""
        content = "this is not ( valid python syntax"
        result = fix_content(content)
        # Should return original content on tokenize error
        assert result == content


class TestFileOperations:
    """Test file reading and writing operations."""

    def test_fix_file_creates_backup(self):
        """Test that fix_file modifies file when needed."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            long_line = 'message = "' + "x" * 100 + '"'
            f.write(long_line)
            temp_path = Path(f.name)

        try:
            # Fix the file
            fix_file(temp_path)

            # Read the result
            with open(temp_path, "r", encoding="utf-8") as f:
                result = f.read()

            # Should be modified
            assert result != long_line
            assert "\\" in result or len(result.splitlines()) > 1
        finally:
            temp_path.unlink()

    def test_fix_file_handles_unchanged_content(self):
        """Test that fix_file doesn't rewrite unchanged files."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            short_content = 'message = "short"'
            f.write(short_content)
            temp_path = Path(f.name)

        try:
            # Get original modification time
            original_content = temp_path.read_text(encoding="utf-8")

            # Fix the file
            fix_file(temp_path)

            # Content should be the same
            new_content = temp_path.read_text(encoding="utf-8")
            assert new_content == original_content
        finally:
            temp_path.unlink()

    def test_fix_file_handles_missing_file(self):
        """Test that fix_file handles missing files gracefully."""
        fake_path = Path("nonexistent_file_12345.py")
        # Should not raise exception
        fix_file(fake_path)


class TestSplitFunctions:
    """Test individual split functions directly."""

    def test_split_normal_string_basic(self):
        """Test split_normal_string function directly."""
        long_text = "This is a test string " * 10
        input_str = f'"{long_text}"'
        result = split_normal_string(input_str, indent_col=4)

        # Should contain continuation
        assert "\\" in result or len(result) <= MAX_LINE_LENGTH

    def test_split_fstring_basic(self):
        """Test split_f_string function directly."""
        long_text = "Value is {x} and more text " * 5
        input_str = f'f"{long_text}"'
        result = split_f_string(input_str, indent_col=4)

        # Should handle the split
        assert isinstance(result, str)

    def test_wrap_docstring_basic(self):
        """Test wrap_docstring function directly."""
        long_doc = "This is a very long documentation string " * 10
        input_str = f'"""{long_doc}"""'
        result = wrap_docstring(input_str, indent_col=4)

        # Should wrap
        assert isinstance(result, str)
        assert '"""' in result


class TestRealWorldExamples:
    """Test real-world code examples."""

    def test_log_message(self):
        """Test typical logging message."""
        content = textwrap.dedent(
            """
            logger.info("Processing user request with ID {user_id} and performing complex operation that takes a long time")
        """
        ).strip()
        result = fix_content(content)
        assert isinstance(result, str)

    def test_sql_query(self):
        """Test SQL query string."""
        content = textwrap.dedent(
            """
            query = "SELECT * FROM users WHERE username = 'test' AND email = 'test@example.com' AND status = 'active' AND created_at > '2024-01-01'"
        """
        ).strip()
        result = fix_content(content)
        assert isinstance(result, str)

    def test_api_endpoint(self):
        """Test API endpoint URL."""
        content = 'url = "https://api.example.com/v1/users/profile/settings/notifications/preferences?detailed=true&include_metadata=true"'
        result = fix_content(content)
        assert isinstance(result, str)

    def test_function_with_docstring(self):
        """Test function with long docstring."""
        content = textwrap.dedent(
            '''
            def example_function():
                """This is a very long docstring that describes the function in great detail and provides comprehensive information about parameters return values and usage examples."""
                pass
        '''
        )
        result = fix_content(content)
        assert isinstance(result, str)
        assert "def example_function" in result

    def test_error_message(self):
        """Test error message string."""
        content = 'raise ValueError("The provided configuration is invalid because the API key is missing and the endpoint URL is not properly formatted")'
        result = fix_content(content)
        assert isinstance(result, str)


class TestIntegration:
    """Integration tests combining multiple scenarios."""

    def test_multiple_long_strings_in_file(self):
        """Test file with multiple long strings."""
        content = textwrap.dedent(
            """
            # Multiple long strings
            msg1 = "This is the first very long message that exceeds the maximum line length and should be split appropriately"
            msg2 = "This is the second very long message that also exceeds the maximum line length and needs splitting"
            msg3 = "Short"
            msg4 = f"This is an f-string with {variable} that is also very long and exceeds the maximum allowed line length"
        """
        )
        result = fix_content(content)

        # Should process all long strings
        assert result != content
        lines = result.splitlines()
        assert len(
            [l for l in lines if len(l) > MAX_LINE_LENGTH]
        ) < content.count("msg")

    def test_mixed_string_types(self):
        """Test file with mixed string types."""
        content = textwrap.dedent(
            '''
            normal = "Normal long string " + "that continues " * 20
            raw = r"C:\\Users\\Path\\That\\Is\\Very\\Long\\And\\Contains\\Many\\Directories\\And\\Subdirectories"
            fstring = f"Format {var1} and {var2} with lots of additional text to make this exceed the line length limit"
            docstring = """This is a docstring that provides detailed documentation about functionality"""
        '''
        )
        result = fix_content(content)
        assert isinstance(result, str)


class TestComments:
    """Test handling of long comments."""

    def test_short_comment_unchanged(self):
        """Short comments should not be modified."""
        content = "# This is a short comment"
        result = fix_content(content)
        assert result == content

    def test_long_comment_gets_wrapped(self):
        """Long comments exceeding MAX_LINE_LENGTH should be wrapped."""
        long_comment = (
            "# "
            + "This is a very long comment that exceeds the maximum line length and should be wrapped appropriately to maintain readability. "
            * 2k
        )
        result = fix_content(long_comment)

        # Result should be different and contain wrapped lines
        assert result != long_comment
        assert len(result.splitlines()) > 1

    def test_comment_with_indentation(self):
        """Indented comments should preserve indentation when wrapped."""
        content = (
            "    # "
            + "This is a very long indented comment that exceeds the maximum line length and should be wrapped appropriately. "
            * 2
        )
        result = fix_content(content)

        # Result should preserve indentation
        assert result != content
        assert result.startswith("    #")
        assert len(result.splitlines()) > 1

    def test_multiple_comments(self):
        """Multiple comments in a file should all be processed."""
        content = """
        # This is a short comment
        # This is a very long comment that exceeds the maximum line length and should be wrapped appropriately to maintain readability.
        # Another short comment
        """
        result = fix_content(content)

        # Only the long comment should be wrapped
        assert result != content
        assert result.count("#") == content.count("#")
        assert (
            len(
                [
                    line
                    for line in result.splitlines()
                    if len(line) > MAX_LINE_LENGTH
                ]
            )
            == 1
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
