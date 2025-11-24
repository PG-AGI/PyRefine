#!/usr/bin/env python3
"""
Command-line interface for string_fixer module.

Usage:
    python cli_string_fixer.py <file_or_directory> [options]

Examples:
    python cli_string_fixer.py myfile.py
    python cli_string_fixer.py src/ --recursive
    python cli_string_fixer.py . -r --line-length 88
    python cli_string_fixer.py myproject/ --dry-run --verbose
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from string_fixer import fix_file, FixStats, MAX_LINE_LENGTH


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def find_python_files(path: Path, recursive: bool = False) -> List[Path]:
    """Find all Python files in the given path.
    
    Args:
        path: Path to search
        recursive: Whether to search recursively
    
    Returns:
        List of Python file paths
    """
    if path.is_file():
        if path.suffix == '.py':
            return [path]
        else:
            logging.warning(f"Not a Python file: {path}")
            return []
    
    if path.is_dir():
        if recursive:
            return list(path.rglob('*.py'))
        else:
            return list(path.glob('*.py'))
    
    return []


def process_files(files: List[Path], dry_run: bool = False,
                  line_length: int = MAX_LINE_LENGTH) -> FixStats:
    """Process multiple Python files.
    
    Args:
        files: List of file paths to process
        dry_run: If True, only report what would be changed
        line_length: Maximum line length to enforce
    
    Returns:
        FixStats object with processing statistics
    """
    stats = FixStats()
    
    # Temporarily update MAX_LINE_LENGTH if specified
    import string_fixer
    original_max_length = string_fixer.MAX_LINE_LENGTH
    if line_length != original_max_length:
        string_fixer.MAX_LINE_LENGTH = line_length
        logging.info(f"Using line length: {line_length}")
    
    try:
        for file_path in files:
            logging.info(f"Processing: {file_path}")
            
            if dry_run:
                # Read and check what would change
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original = f.read()
                    
                    from string_fixer import fix_content
                    fixed = fix_content(original, stats=stats)
                    
                    stats.files_processed += 1
                    if original != fixed:
                        stats.files_modified += 1
                        logging.info(f"  Would modify: {file_path}")
                    else:
                        logging.debug(f"  No changes: {file_path}")
                        
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {e}"
                    logging.error(error_msg)
                    stats.add_error(error_msg)
            else:
                # Actually fix the file
                was_modified = fix_file(file_path, stats=stats)
                if was_modified:
                    logging.info(f"  Modified: {file_path}")
                else:
                    logging.debug(f"  No changes: {file_path}")
    
    finally:
        # Restore original MAX_LINE_LENGTH
        string_fixer.MAX_LINE_LENGTH = original_max_length
    
    return stats


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Fix long Python strings by splitting them.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s myfile.py                    # Fix a single file
  %(prog)s src/ -r                      # Fix all .py files recursively
  %(prog)s . --dry-run --verbose        # Preview changes
  %(prog)s myproject/ -r -l 88          # Use 88 char length
        """
    )
    
    parser.add_argument(
        'path',
        type=Path,
        help='File or directory to process'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Process directories recursively'
    )
    
    parser.add_argument(
        '-d', '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '-l', '--line-length',
        type=int,
        default=MAX_LINE_LENGTH,
        help=f'Maximum line length (default: {MAX_LINE_LENGTH})'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create .bak backup files'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate path
    if not args.path.exists():
        logging.error(f"Path does not exist: {args.path}")
        sys.exit(1)
    
    # Find files to process
    logging.info(f"Scanning for Python files in: {args.path}")
    files = find_python_files(args.path, args.recursive)
    
    if not files:
        logging.warning("No Python files found")
        sys.exit(0)
    
    logging.info(f"Found {len(files)} Python file(s)")
    
    if args.dry_run:
        logging.info("DRY RUN MODE - No files will be modified")
    
    # Process files
    stats = process_files(
        files, dry_run=args.dry_run, line_length=args.line_length
    )
    
    # Print statistics
    print("\n" + "="*50)
    print("STRING FIXER RESULTS")
    print("="*50)
    print(stats)
    
    if args.dry_run and stats.files_modified > 0:
        print("\nRun without --dry-run to apply changes")
    
    # Exit with appropriate code
    if stats.errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
