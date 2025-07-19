import re
import logging
from typing import Dict, List, TypedDict

logging.basicConfig(level=logging.INFO)

class Patch(TypedDict):
    file_path: str
    diff: str

class UnitTest(TypedDict):
    file_path: str
    code: str

def parse_llm_output(text: str) -> Dict[str, List[Patch] | List[UnitTest]]:
    """
    Parses an LLM-generated response to extract code patches and unit tests.

    Args:
        text: The LLM-generated text.

    Returns:
        A dictionary with two keys:
        - "patches": A list of dictionaries, where each dictionary represents a
          unified diff patch and contains the "file_path" and "diff" text.
        - "unit_tests": A list of dictionaries, where each dictionary
          represents a unit test and contains the "file_path" and "code".
    """
    patches = _extract_patches(text)
    unit_tests = _extract_unit_tests(text)

    if not patches:
        logging.info("No patches found in the response.")
    if not unit_tests:
        logging.info("No unit tests found in the response.")

    return {"patches": patches, "unit_tests": unit_tests}

def _validate_patch(diff_text: str) -> bool:
    """
    Validates the patch to prevent basic TypeErrors.
    This is a simplified check. A more robust solution might involve a static analysis tool.
    """
    # Avoid adding a string and an integer
    if re.search(r"\+\s*.*(\w+\s*\+\s*['\"].*['\"])", diff_text) or re.search(r"\+\s*.*(['\"].*['\"]\s*\+\s*\w+)", diff_text):
        # Check if it's a string concatenation
        if not re.search(r"\+\s*.*(['\"].*['\"]\s*\+\s*['\"].*['\"])", diff_text):
             logging.warning(f"Potential TypeError detected in patch:\n{diff_text}")
             return False
    return True

def _extract_patches(text: str) -> List[Patch]:
    """Extracts unified diff patches from the text."""
    # Regex to find unified diff blocks
    diff_pattern = re.compile(
        r"diff --git a/(.+) b/(.+)\n--- a/.*\n\+\+\+ b/.*\n@@ .* @@\n([\s\S]*?)(?=\ndiff --git|\Z)",
        re.MULTILINE,
    )
    patches = []
    for match in diff_pattern.finditer(text):
        file_path = match.group(1)
        # Reconstruct the full diff text
        diff_text = f"diff --git a/{file_path} b/{match.group(2)}\n--- a/{file_path}\n+++ b/{match.group(2)}\n@@ {match.group(3)}"
        if _validate_patch(diff_text):
            patches.append({"file_path": file_path, "diff": diff_text})
            logging.info(f"Found and validated patch for file: {file_path}")
        else:
            logging.warning(f"Invalid patch detected for file {file_path}. Skipping.")
            
    return patches


def _extract_file_path(text: str, match_start: int) -> str | None:
    """Extracts the file path for a given match."""
    preceding_text = text[:match_start]
    
    # 1. Look for diff headers
    diff_header_match = re.findall(r"diff --git a/(\S+) b/\S+", preceding_text)
    if diff_header_match:
        return diff_header_match[-1]

    # 2. Look for --- a/ or +++ b/
    plus_minus_header_match = re.findall(r"--- a/(\S+)", preceding_text)
    if plus_minus_header_match:
        return plus_minus_header_match[-1]
    
    plus_header_match = re.findall(r"\+\+\+ b/(\S+)", preceding_text)
    if plus_header_match:
        return plus_header_match[-1]

    # 3. Fallback to comment lines or markers
    marker_match = re.findall(r"(?:#|File:|Path:)\s*([\w/\\-]+\.py)", preceding_text)
    if marker_match:
        path = marker_match[-1]
        return path.replace("\\", "/").strip()

    logging.warning("Could not determine file path for a test block.")
    return None


def _extract_unit_tests(text: str) -> List[UnitTest]:
    """Extracts Python unit tests from the text."""
    unit_tests: List[UnitTest] = []
    processed_tests = set()

    # Extract from Python code fences
    fenced_code_pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    for match in fenced_code_pattern.finditer(text):
        code_block = match.group(1)
        test_funcs = re.findall(r"def (test_[A-Za-z0-9_]+)", code_block)
        if test_funcs:
            file_path = _extract_file_path(text, match.start())
            for func in test_funcs:
                if func not in processed_tests:
                    unit_tests.append({"file_path": file_path, "code": code_block})
                    processed_tests.add(func)
                    logging.info(f"Found fended unit test for file: {file_path}")
                    break

    # Extract inline test definitions
    inline_test_pattern = re.compile(r"^(def (test_[A-Za-z0-9_]+)\(.*?\):.*?)(?=\n\n|\Z)", re.DOTALL | re.MULTILINE)
    for match in inline_test_pattern.finditer(text):
        test_code = match.group(1)
        test_func = match.group(2)
        if test_func not in processed_tests:
            file_path = _extract_file_path(text, match.start())
            unit_tests.append({"file_path": file_path, "code": test_code})
            processed_tests.add(test_func)
            logging.info(f"Found inline unit test for file: {file_path}")

    return unit_tests