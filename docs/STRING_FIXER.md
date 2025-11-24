# String Fixer Documentation

## Overview

`string_fixer.py` is a Python code formatting tool that automatically fixes long string lines (exceeding 79 characters by default) by intelligently splitting them. It helps maintain PEP 8 compliance and improves code readability.

## What It Does

The string fixer analyzes Python source code and automatically splits long strings into multiple lines using continuation backslashes. It intelligently handles different types of strings:

1. **Normal Strings** - Regular `"..."` or `'...'` strings
2. **Raw Strings** - `r"..."` strings (commonly used for paths and regex)
3. **F-Strings** - `f"..."` formatted strings with expressions
4. **Docstrings** - Triple-quoted `"""..."""` documentation strings

## Use Cases

### ✅ Scenarios It Solves

1. **Long Log Messages**
   ```python
   # Before
   logger.info("Processing user request with ID 12345 and performing complex operation that takes a long time to complete")
   
   # After
   logger.info("Processing user request with ID 12345 and performing " \
               "complex operation that takes a long time to complete")
   ```

2. **SQL Queries**
   ```python
   # Before
   query = "SELECT * FROM users WHERE username = 'test' AND email = 'test@example.com' AND status = 'active'"
   
   # After
   query = "SELECT * FROM users WHERE username = 'test' AND " \
           "email = 'test@example.com' AND status = 'active'"
   ```

3. **API URLs**
   ```python
   # Before
   url = "https://api.example.com/v1/users/profile/settings/notifications/preferences?detailed=true"
   
   # After
   url = "https://api.example.com/v1/users/profile/settings/" \
         "notifications/preferences?detailed=true"
   ```

4. **Error Messages**
   ```python
   # Before
   raise ValueError("The provided configuration is invalid because the API key is missing")
   
   # After
   raise ValueError("The provided configuration is invalid " \
                    "because the API key is missing")
   ```

5. **Long Docstrings**
   ```python
   # Before
   """This is a very long docstring that describes the function in great detail and provides comprehensive information."""
   
   # After
   """This is a very long docstring that describes the function in great
   detail and provides comprehensive information."""
   ```

### ❌ Scenarios It CAN'T Handle

1. **Comments** - Does not wrap long comments
2. **Import Statements** - Does not split long imports
3. **URLs That Shouldn't Break** - May break URLs at spaces (semantic issue)
4. **Multi-line Strings** - Already-split strings remain unchanged
5. **Complex Nested F-Strings** - May have issues with deeply nested expressions
6. **Code Logic** - Does not detect if splitting changes semantic meaning

## Features

### Core Functionality

- ✅ Tokenization-based parsing (accurate and safe)
- ✅ Intelligent split points (prefers word boundaries)
- ✅ Preserves escape sequences
- ✅ Handles indentation correctly
- ✅ F-string expression preservation
- ✅ Configurable line length
- ✅ Statistics tracking
- ✅ Error handling and logging
- ✅ Backup file creation

### Statistics Tracking

The `FixStats` class tracks:
- Files processed
- Files modified
- Strings fixed
- Lines reduced/added
- Errors encountered

### Error Handling

- Binary file detection
- Unicode encoding errors
- Permission errors
- Missing file handling
- Tokenization errors
- Graceful degradation

## Usage

### As a Module

```python
from pathlib import Path
from string_fixer import fix_file, fix_content, FixStats

# Fix a single file
fix_file(Path("myfile.py"))

# Fix content directly
content = 'msg = "very long string..."'
fixed = fix_content(content)

# With statistics tracking
stats = FixStats()
fix_file(Path("myfile.py"), stats=stats)
print(stats)
```

### As a CLI Tool

```bash
# Fix a single file
python cli_string_fixer.py myfile.py

# Fix all Python files in a directory recursively
python cli_string_fixer.py src/ --recursive

# Dry run to preview changes
python cli_string_fixer.py . --dry-run --verbose

# Use Black's 88-character line length
python cli_string_fixer.py myproject/ -r --line-length 88

# Show help
python cli_string_fixer.py --help
```

### CLI Options

- `path` - File or directory to process
- `-r, --recursive` - Process directories recursively
- `-d, --dry-run` - Preview changes without modifying files
- `-v, --verbose` - Enable detailed output
- `-l, --line-length` - Set maximum line length (default: 79)
- `--no-backup` - Skip backup file creation

## Testing

Comprehensive test suite in `tests/test_string_fixer.py` covers:

### Test Categories

1. **Normal Strings** - Single/double quoted strings
2. **Raw Strings** - Windows paths, regex patterns
3. **F-Strings** - Variable interpolation, expressions
4. **Docstrings** - Triple-quoted documentation
5. **Edge Cases** - Empty files, syntax errors, encoding issues
6. **File Operations** - Reading, writing, backups
7. **Real-World Examples** - Log messages, SQL, APIs
8. **Integration** - Multiple strings, mixed types

### Running Tests

```bash
# Run all tests
pytest tests/test_string_fixer.py -v

# Run specific test class
pytest tests/test_string_fixer.py::TestNormalStrings -v

# Run with coverage
pytest tests/test_string_fixer.py --cov=string_fixer --cov-report=html
```

## Configuration

### Line Length

The default line length is 79 characters (PEP 8 standard). You can customize this:

**In code:**
```python
import string_fixer
string_fixer.MAX_LINE_LENGTH = 88  # Black style
```

**Via CLI:**
```bash
python cli_string_fixer.py myfile.py --line-length 88
```

### Logging

Configure logging level:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with PyRefine

String fixer can be integrated into PyRefine's clean command:

```python
from commands.clean.clean_manager import CleanManager
from string_fixer import fix_file, FixStats

class CleanManager:
    def format_python_files(self, path: Path):
        stats = FixStats()
        # Fix long strings before other formatters
        fix_file(path, stats)
        # Then run autoflake, isort, autopep8, black...
```

## Best Practices

### When to Use

1. **Before Committing** - Ensure all strings comply with line limits
2. **After Code Generation** - Clean up auto-generated code
3. **Refactoring Legacy Code** - Modernize old codebases
4. **CI/CD Pipeline** - Automated enforcement

### When NOT to Use

1. **On External Libraries** - Don't modify third-party code
2. **Generated Files** - May be overwritten
3. **Binary Files** - Use `.gitattributes` to exclude
4. **Minified Code** - Not relevant for minified JavaScript/CSS

## Limitations

1. **Semantic Awareness** - Does not understand if splitting changes meaning
2. **Complex F-Strings** - May struggle with deeply nested expressions
3. **Performance** - Tokenization can be slow on very large files
4. **Comments** - Only handles strings, not comments
5. **Non-Python Files** - Only works with Python source code

## Future Enhancements

Potential improvements:

1. **Comment Wrapping** - Add support for long comments
2. **Import Statement Splitting** - Handle long imports
3. **Configuration File** - `.string_fixer.yml` for project settings
4. **URL Detection** - Avoid breaking URLs at bad points
5. **Custom Split Rules** - User-defined split strategies
6. **IDE Integration** - VS Code extension
7. **Pre-commit Hook** - Git hook integration
8. **Performance** - Parallel processing for large projects

## Examples

### Example 1: Log Message

**Before:**
```python
logger.error("Failed to connect to database server at 192.168.1.100 with credentials user=admin")
```

**After:**
```python
logger.error("Failed to connect to database server at 192.168.1.100 " \
             "with credentials user=admin")
```

### Example 2: F-String

**Before:**
```python
message = f"User {username} with ID {user_id} successfully logged in from IP {ip_address}"
```

**After:**
```python
message = f"User {username} with ID {user_id} successfully logged in " \
          f"from IP {ip_address}"
```

### Example 3: Docstring

**Before:**
```python
def process_data():
    """This function processes user data by validating input, transforming it, and storing in database."""
    pass
```

**After:**
```python
def process_data():
    """This function processes user data by validating input, transforming
    it, and storing in database."""
    pass
```

## Troubleshooting

### Issue: File Not Modified

**Possible Causes:**
- File already compliant with line length
- Tokenization error (syntax error in file)
- File permissions issue

**Solution:** Run with `--verbose` to see details

### Issue: String Split at Wrong Place

**Cause:** Limited lookahead for finding split points

**Solution:** Manually adjust or increase line length limit

### Issue: F-String Broken

**Cause:** Complex expression inside f-string

**Solution:** Simplify expression or split manually

## Contributing

To contribute improvements to string_fixer:

1. Add tests in `tests/test_string_fixer.py`
2. Update this documentation
3. Ensure all tests pass: `pytest tests/test_string_fixer.py`
4. Submit pull request

## License

Part of PyRefine project - see main LICENSE file.
