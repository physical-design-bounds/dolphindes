"""Utilities for file managerment and IO."""


def print_underline(text: str) -> None:
    """Print underlined text to the console.

    Args:
        text: Text to be underlined.
    """
    underline = "-" * len(text)
    print()
    print(f"{text}\n{underline}")
