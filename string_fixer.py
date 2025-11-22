import io
import textwrap
import tokenize
from pathlib import Path
from typing import List, Tuple

MAX_LINE_LENGTH = 79


def fix_file(path: Path) -> None:
    """Reads a file, fixes string lengths, and writes it back."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        fixed_content = fix_content(content)

        if content != fixed_content:
            with open(path, "w", encoding="utf-8") as f:
                f.write(fixed_content)
    except Exception as e:
        print(f"Failed to fix strings in {path}: {e}")


def fix_content(content: str) -> str:
    """Parses content and splits long strings."""
    try:
        tokens = list(
            tokenize.tokenize(io.BytesIO(content.encode("utf-8")).readline)
        )
    except tokenize.TokenError:
        return content

    # Identify ranges to replace and replace them from bottom to top.
    replacements: List[Tuple[int, int, str]] = []

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

            i = j  # Skip processed tokens
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


def split_string_token(
    token: tokenize.TokenInfo, line_content: str, start_col: int
) -> str:
    string_val = token.string

    if string_val.startswith('"""') or string_val.startswith("'''"):
        return wrap_docstring(string_val, start_col)
    elif string_val.startswith('"') or string_val.startswith("'"):
        return split_normal_string(string_val, start_col)
    elif string_val.lower().startswith('f"') or string_val.lower().startswith(
        "f'"
    ):
        return split_f_string(string_val, start_col)
    elif string_val.lower().startswith('r"') or string_val.lower().startswith(
        "r'"
    ):
        return split_normal_string(string_val, start_col)

    return string_val


def split_normal_string(text: str, indent_col: int) -> str:
    if text.startswith("r") or text.startswith("R"):
        prefix = text[:2]
        quote = text[1]
        content = text[2:-1]
    else:
        prefix = text[0]
        quote = text[0]
        content = text[1:-1]

    available = MAX_LINE_LENGTH - indent_col - len(prefix) - 1

    if available <= 0:
        available = 40

    chunks = []
    current = content

    while len(current) > available:
        break_point = current.rfind(" ", 0, available)
        if break_point == -1:
            break_point = available - 1

        # Ensure we don't split at a backslash that would escape the quote
        while break_point > 0 and current[break_point] == "\\":
            bs_count = 0
            idx = break_point
            while idx >= 0 and current[idx] == "\\":
                bs_count += 1
                idx -= 1

            if bs_count % 2 == 1:
                break_point -= 1
            else:
                break

        if break_point < 0:
            break_point = available - 1

        chunk = current[: break_point + 1]
        chunks.append(chunk)
        current = current[break_point + 1:]

        available = MAX_LINE_LENGTH - indent_col - len(prefix) - 1
        if available < 10:
            available = 50

    chunks.append(current)

    result = ""
    for i, chunk in enumerate(chunks):
        if i > 0:
            result += " \\\n" + " " * indent_col
        result += f"{prefix}{chunk}{quote}"

    return result


def split_f_string(text: str, indent_col: int) -> str:
    if text.lower().startswith("fr") or text.lower().startswith("rf"):
        prefix = text[:3]
        quote = text[2]
        content = text[3:-1]
    else:
        prefix = text[:2]
        quote = text[1]
        content = text[2:-1]

    available = MAX_LINE_LENGTH - indent_col - len(prefix) - 1

    chunks = []
    current = content

    while len(current) > available:
        break_point = -1
        depth = 0
        scan_limit = min(len(current), available)
        last_space = -1

        for i in range(scan_limit):
            char = current[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            elif char == " " and depth == 0:
                last_space = i

        if last_space != -1:
            break_point = last_space
        else:
            break

        chunk = current[: break_point + 1]
        chunks.append(chunk)
        current = current[break_point + 1:]
        available = MAX_LINE_LENGTH - indent_col - len(prefix) - 1

    chunks.append(current)

    if len(chunks) == 1:
        return text

    result = ""
    for i, chunk in enumerate(chunks):
        if i > 0:
            result += " \\\n" + " " * indent_col
        result += f"{prefix}{chunk}{quote}"
    return result


def wrap_docstring(text: str, indent_col: int) -> str:
    quote = text[:3]
    content = text[3:-3]

    width = MAX_LINE_LENGTH - indent_col - len(quote)
    if width < 20:
        width = 79

    wrapped = textwrap.fill(content, width=width)
    lines = wrapped.splitlines()

    new_content = ""
    for i, line in enumerate(lines):
        if i > 0:
            new_content += "\n" + " " * indent_col
        new_content += line

    return f"{quote}{new_content}{quote}"
