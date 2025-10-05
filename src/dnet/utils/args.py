"""Argument parsing utilities for dnet."""

import argparse
import ast
from typing import List


def string_to_list(s: str) -> List:
    """Convert string input to a list using ast.literal_eval.

    Args:
        s: String representation of a list (e.g., "[1, 2, 3]")

    Returns:
        Parsed list

    Raises:
        argparse.ArgumentTypeError: If input is not a valid list
    """
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
        else:
            raise ValueError("Not a list")
    except Exception:
        raise argparse.ArgumentTypeError("Input must be a valid list.")


def string_to_nested_list(s: str) -> List[List]:
    """Convert string input to a nested list using ast.literal_eval.

    Args:
        s: String representation of a nested list (e.g., "[[1, 2], [3, 4]]")

    Returns:
        Parsed nested list

    Raises:
        argparse.ArgumentTypeError: If input is not a valid nested list
    """
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            if all(isinstance(item, list) for item in result):
                return result
            else:
                raise ValueError(
                    "Not all elements are lists - this is not a proper nested list"
                )
        else:
            raise ValueError("Not a list")
    except Exception:
        raise argparse.ArgumentTypeError("Input must be a valid nested list.")


def is_all_ints(lst: List) -> bool:
    """Check if a list (potentially nested) contains only integers.

    Args:
        lst: List to check

    Returns:
        True if all elements are integers, False otherwise
    """

    def helper(lst):
        if isinstance(lst, int):
            return True
        if isinstance(lst, list):
            return all(helper(x) for x in lst)
        return False

    return isinstance(lst, list) and helper(lst)


def string_to_ilist(s: str) -> List[int]:
    """Convert string to list of integers.

    Args:
        s: String representation of integer list (e.g., "[1, 2, 3]")

    Returns:
        Parsed list of integers

    Raises:
        ValueError: If input contains non-integers
    """
    result = string_to_list(s)
    if not is_all_ints(result):
        raise ValueError("Input must only contain integers")
    return result


def string_to_nested_ilist(s: str) -> List[List[int]]:
    """Convert string to nested list of integers.

    Args:
        s: String representation of nested integer list (e.g., "[[1, 2], [3, 4]]")

    Returns:
        Parsed nested list of integers

    Raises:
        ValueError: If input contains non-integers
    """
    result = string_to_nested_list(s)
    if not is_all_ints(result):
        raise ValueError("Input must only contain integers")
    return result
