import io
import logging
import textwrap
import tokenize
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

MAX_LINE_LENGTH = 79

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FixStats:
    """Statistics about string fixing operations."""

    files_processed: int = 0
    files_modified: int = 0
    strings_fixed: int = 0
    lines_reduced: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def add_error(self, error: str):
        """Add an error message to the stats."""
        self.errors.append(error)

    def __str__(self) -> str:
        """Return a formatted string representation of stats."""
        output = []
        output.append(f"Files processed: {self.files_processed}")
        output.append(f"Files modified: {self.files_modified}")
        output.append(f"Strings fixed: {self.strings_fixed}")
        output.append(f"Lines reduced: {self.lines_reduced}")
        if self.errors:
            output.append(f"Errors encountered: {len(self.errors)}")
            for error in self.errors[:5]:  # Show first 5 errors
                output.append(f"  - {error}")
        return "\n".join(output)


def fix_file(
    path: Path,
    stats: Optional[FixStats] = None,
    *,
    create_backup: bool = True,
) -> bool:
    """Reads a file, fixes string lengths, and writes it back.

    Args:
        path: Path to the file to fix
        stats: Optional FixStats object to track statistics
        create_backup: whether to create .bak before writing

    Returns:
        True if file was modified, False otherwise
    """
    if stats:
        stats.files_processed += 1

    try:
        # Check if file exists
        if not path.exists():
            error_msg = f"File not found: {path}"
            logger.error(error_msg)
            if stats:
                stats.add_error(error_msg)
            return False

        # Check if file is binary
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            error_msg = f"Cannot process binary file: {path}"
            logger.warning(error_msg)
            if stats:
                stats.add_error(error_msg)
            return False

        # Check if file is empty
        if not content.strip():
            logger.debug(f"Skipping empty file: {path}")
            return False

        original_line_count = len(content.splitlines())
        fixed_content = fix_content(content, stats=stats)

        if content != fixed_content:
            new_line_count = len(fixed_content.splitlines())

            # Create backup
            if create_backup:
                backup_path = path.with_suffix(path.suffix + ".bak")
                try:
                    with open(backup_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.debug(f"Created backup: {backup_path}")
                except Exception as e:
                    logger.warning(f"Could not create backup: {e}")

            # Write fixed content
            with open(path, "w", encoding="utf-8") as f:
                f.write(fixed_content)

            if stats:
                stats.files_modified += 1
                # positive number means we reduced lines
                stats.lines_reduced += original_line_count - new_line_count

            logger.info(f"Fixed strings in {path}")
            return True

        logger.debug(f"No changes needed for {path}")
        return False

    except PermissionError as e:
        error_msg = f"Permission denied: {path} - {e}"
        logger.error(error_msg)
        if stats:
            stats.add_error(error_msg)
        return False
    except Exception as e:
        error_msg = f"Failed to fix strings in {path}: {e}"
        logger.error(error_msg)
        if stats:
            stats.add_error(error_msg)
        return False


def fix_content(content: str, stats: Optional[FixStats] = None) -> str:
    """Parses content and splits long strings and comments."""
    try:
        tokens = list(
            tokenize.tokenize(io.BytesIO(content.encode("utf-8")).readline)
        )
    except tokenize.TokenError as e:
        logger.debug(f"Tokenize error: {e}")
        return content
    except Exception as e:
        logger.warning(f"Unexpected error during tokenization: {e}")
        return content

    # Identify ranges to replace and replace them from bottom to top.
    # (start_line, start_col, end_line, end_col, new_text)
    replacements: List[Tuple[int, int, int, int, str]] = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.type == tokenize.STRING:
            start_line, start_col = token.start
            end_line, end_col = token.end
            string_val = token.string

            if start_line == end_line:
                line_content = content.splitlines(keepends=True)[
                    start_line - 1
                ]
                if len(line_content) > MAX_LINE_LENGTH:
                    new_string = split_string_token(
                        token, line_content, start_col
                    )
                    if new_string != string_val:
                        replacements.append(
                            (
                                start_line,
                                start_col,
                                end_line,
                                end_col,
                                new_string,
                            )
                        )
                        if stats:
                            stats.strings_fixed += 1
            i += 1

        elif token.type == getattr(tokenize, "FSTRING_START", -1):
            # Found start of f-string. Find the end.
            start_token = token
            balance = 1
            j = i + 1
            while j < len(tokens) and balance > 0:
                t = tokens[j]
                if t.type == getattr(tokenize, "FSTRING_START", -1):
                    balance += 1
                elif t.type == getattr(tokenize, "FSTRING_END", -1):
                    balance -= 1
                j += 1

            end_token = tokens[j - 1]
            start_line, start_col = start_token.start
            end_line, end_col = end_token.end

            if start_line == end_line:
                line_content = content.splitlines(keepends=True)[
                    start_line - 1
                ]
                if len(line_content) > MAX_LINE_LENGTH:
                    raw_f_string = line_content[start_col:end_col]
                    new_string = split_f_string(raw_f_string, start_col)
                    if new_string != raw_f_string:
                        replacements.append(
                            (
                                start_line,
                                start_col,
                                end_line,
                                end_col,
                                new_string,
                            )
                        )
                        if stats:
                            stats.strings_fixed += 1

            i = j  # Skip processed tokens

        elif token.type == tokenize.COMMENT:
            start_line, start_col = token.start
            comment_val = token.string

            # consider indentation in visual length
            if start_col + len(comment_val) > MAX_LINE_LENGTH:
                new_comment = wrap_comment(comment_val, start_col)
                if new_comment != comment_val:
                    replacements.append(
                        (
                            start_line,
                            start_col,
                            start_line,
                            start_col + len(comment_val),
                            new_comment,
                        )
                    )
                    if stats:
                        stats.strings_fixed += 1

            i += 1
        else:
            i += 1

    # Apply replacements from last to first
    replacements.sort(key=lambda x: (x[0], x[1]), reverse=True)

    lines = content.splitlines(keepends=True)

    for start_line, start_col, end_line, end_col, new_text in replacements:
        original_line = lines[start_line - 1]
        prefix = original_line[:start_col]
        suffix = original_line[end_col:]
        new_lines_text = prefix + new_text + suffix
        lines[start_line - 1] = new_lines_text

    return "".join(lines)


# ---------- String literal helpers/splitters ----------

_STRING_PREFIX_RE = re.compile(r'(?i)^([rubf]{1,3})?([\'"]{3}|[\'"])')


def _split_literal(text: str) -> Tuple[str, str, str]:
    # """
    # Returns (prefix, quote, body) for a Python string literal like:
    #   r"abc", fr'xyz', """doc""", r'''doc''', "plain"
    # prefix: e.g., 'r', 'f', 'fr', '', 'b', 'rf'
    # quote: one of: '"', "'", '"""', "'''"
    # body: inner content (without the closing quote)
    # """
    m = _STRING_PREFIX_RE.match(text)
    if not m:
        # Fallback: treat whole as body
        return "", "", text
    prefix = m.group(1) or ""
    quote = m.group(2)
    # find the matching closing quote from the end
    end = text.rfind(quote)
    if end == -1:
        end = len(text)
    body = text[m.end() : end]
    return prefix, quote, body


def split_string_token(
    token: tokenize.TokenInfo, line_content: str, start_col: int
) -> str:
    text = token.string
    prefix, quote, body = _split_literal(text)

    # Triple-quoted (incl. with prefixes like r/f/rf) -> treat as docstring-like
    if len(quote) == 3:
        return wrap_docstring(text, start_col)

    # f-strings (including rf, fr)
    if "f" in prefix.lower():
        return split_f_string(text, start_col)

    # Everything else (r, b, u, rb, br, or plain)
    return split_normal_string(text, start_col)


def split_normal_string(text: str, indent_col: int) -> str:
    prefix, quote, content = _split_literal(text)

    available = MAX_LINE_LENGTH - indent_col - len(prefix) - len(quote)
    if available <= 0:
        available = 40

    chunks: List[str] = []
    current = content

    while len(current) > available:
        break_point = current.rfind(" ", 0, available)
        if break_point == -1:
            break_point = max(0, available - 1)

        # Ensure we don't end chunk with an odd number of backslashes
        bp = break_point
        while bp > 0 and current[bp] == "\\":
            # count trailing backslashes up to bp
            k = bp
            bs = 0
            while k >= 0 and current[k] == "\\":
                bs += 1
                k -= 1
            if bs % 2 == 1:
                bp -= 1
            else:
                break
        break_point = max(0, bp)

        chunk = current[: break_point + 1]

        # raw strings cannot end with a single backslash
        if "r" in prefix.lower() and chunk.endswith("\\"):
            if len(current) > break_point + 1:
                chunk += current[break_point + 1]
                current = current[break_point + 2 :]
            else:
                # can't fix safely; stop splitting
                chunks.append(current)
                current = ""
                break
        else:
            current = current[break_point + 1 :]

        chunks.append(chunk)

        available = MAX_LINE_LENGTH - indent_col - len(prefix) - len(quote)
        if available < 10:
            available = 50

    if current:
        chunks.append(current)

    if len(chunks) == 1:
        return text

    result = ""
    for i, chunk in enumerate(chunks):
        if i > 0:
            result += " \\\n" + " " * indent_col
        result += f"{prefix}{quote}{chunk}{quote}"
    return result


def split_f_string(text: str, indent_col: int) -> str:
    prefix, quote, content = _split_literal(text)
    available = MAX_LINE_LENGTH - indent_col - len(prefix) - len(quote)

    chunks: List[str] = []
    current = content

    while len(current) > available:
        break_point = -1
        depth = 0
        last_space = -1
        scan_limit = min(len(current), available)

        for i, ch in enumerate(current[:scan_limit]):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == " " and depth == 0:
                last_space = i

        if last_space == -1:
            break

        break_point = last_space
        chunk = current[: break_point + 1]
        chunks.append(chunk)
        current = current[break_point + 1 :]

        available = MAX_LINE_LENGTH - indent_col - len(prefix) - len(quote)

    if current:
        chunks.append(current)

    if len(chunks) == 1:
        return text

    result = ""
    for i, chunk in enumerate(chunks):
        if i > 0:
            result += " \\\n" + " " * indent_col
        result += f"{prefix}{quote}{chunk}{quote}"
    return result


def wrap_docstring(text: str, indent_col: int) -> str:
    # support prefixes like r"""..."""
    prefix, quote, content = _split_literal(text)
    if len(quote) != 3:
        # not triple, defer to normal behavior
        return text

    width = MAX_LINE_LENGTH - indent_col - len(prefix) - len(quote)
    if width < 20:
        width = 79

    wrapped = textwrap.fill(content, width=width)
    lines = wrapped.splitlines()

    new_content = ""
    for i, line in enumerate(lines):
        if i > 0:
            new_content += "\n" + " " * indent_col
        new_content += line

    return f"{prefix}{quote}{new_content}{quote}"


def wrap_comment(comment: str, indent_col: int) -> str:
    """Wraps a long comment into multiple lines, preserving indentation."""
    prefix = "# "
    # Remove exactly one leading '#' (and one optional space), keep the rest
    if comment.startswith("#"):
        inner = comment[1:].lstrip()
    else:
        inner = comment.lstrip()

    # Calculate available width for wrapping
    width = MAX_LINE_LENGTH - indent_col - len(prefix)
    if width < 20:
        width = 79

    # Wrap the comment content
    wrapped = textwrap.fill(inner, width=width)
    lines = wrapped.splitlines()

    # Add indentation and prefix to each line
    new_comment = "\n".join(
        (" " * indent_col + prefix + line) for line in lines
    )

    return new_comment
