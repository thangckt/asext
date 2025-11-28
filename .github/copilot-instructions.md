# Project Overview
ASEXT: Python package extends functions of ASE (Atomic Simulation Environment).

## Development Notes
- Always add type hints - Include proper type annotations in all Python code for better maintainability, except in test files. Functions returning `None` should omit return type hints.
- Always use conventional commit format - All commit messages and PR titles must follow the conventional commit specification (e.g., `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
- Extensive examples - Use `example/` directory as reference for proper configuration formats
- Ignore any commented-out code when reviewing or suggesting changes.


## Excluded Files
Do not review or suggest changes for the following files and directories:
- "1devtools/**"
- "*.ipynb"
