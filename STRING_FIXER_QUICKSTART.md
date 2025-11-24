# String Fixer - Quick Start Guide

## 🚀 What Is This?

A tool that automatically fixes long Python strings (>79 characters) by splitting them into multiple lines.

## ⚡ Quick Usage

### Fix a single file:
```bash
python cli_string_fixer.py myfile.py
```

### Fix all Python files in a directory:
```bash
python cli_string_fixer.py src/ --recursive
```

### Preview changes without modifying files:
```bash
python cli_string_fixer.py . --dry-run --verbose
```

### Use Black's line length (88 chars):
```bash
python cli_string_fixer.py myfile.py --line-length 88
```

## 📋 What Gets Fixed

### Before:
```python
error_message = "This is a very long error message that exceeds the maximum line length limit"
```

### After:
```python
error_message = "This is a very long error message that exceeds the " \
                "maximum line length limit"
```

## 🎯 Handles These String Types

- ✅ Normal strings: `"text"`
- ✅ Raw strings: `r"C:\path"`
- ✅ F-strings: `f"Hello {name}"`
- ✅ Docstrings: `"""docs"""`
- ✅ SQL queries
- ✅ API URLs
- ✅ Log messages

## 🧪 Run Tests

```bash
pytest tests/test_string_fixer.py -v
```

**Result:** 34 tests, all passing ✓

## 📖 Full Documentation

See `docs/STRING_FIXER.md` for complete documentation.

## 💡 Examples

### Example 1: Fix SQL queries
```bash
python cli_string_fixer.py database.py
```

### Example 2: Fix entire project
```bash
python cli_string_fixer.py . -r --verbose
```

### Example 3: Check what would change
```bash
python cli_string_fixer.py myproject/ -r --dry-run
```

## 🔧 Options

| Option | Description |
|--------|-------------|
| `--recursive` / `-r` | Process directories recursively |
| `--dry-run` / `-d` | Preview without modifying |
| `--verbose` / `-v` | Show detailed output |
| `--line-length` / `-l` | Set max line length (default: 79) |

## 📊 Output Example

```
==================================================
STRING FIXER RESULTS
==================================================
Files processed: 15
Files modified: 8
Strings fixed: 42
Lines reduced: 73
```

## ⚠️ Safety Features

- Creates `.bak` backup files
- Handles encoding errors gracefully
- Skips binary files automatically
- Reports errors without crashing

## 🎓 Try It Now!

```bash
# 1. Create a test file with long strings
echo 'msg = "This is a very long message that definitely exceeds the line limit"' > test.py

# 2. Fix it
python cli_string_fixer.py test.py --verbose

# 3. Check the result
cat test.py
```

## 📞 Need Help?

- Full docs: `docs/STRING_FIXER.md`
- Summary: `docs/STRING_FIXER_SUMMARY.md`
- Tests: `tests/test_string_fixer.py`
- Demo: `demo_long_strings.py`
