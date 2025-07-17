import unittest
from patch_parser import parse_llm_output

class TestPatchParser(unittest.TestCase):

    def test_standard_diff_and_fenced_tests(self):
        text = """
Some text about the changes.

diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,5 +1,5 @@
 def hello():
-    print("Hello, world!")
+    print("Hello, new world!")

 def goodbye():
     print("Goodbye, world!")
```python
import unittest
from main import hello

class TestMain(unittest.TestCase):
    def test_hello(self):
        # ... test implementation ...
        pass
```
"""
        result = parse_llm_output(text)
        self.assertEqual(len(result["patches"]), 1)
        self.assertEqual(result["patches"][0]["file_path"], "src/main.py")
        self.assertEqual(len(result["unit_tests"]), 1)
        self.assertEqual(result["unit_tests"][0]["file_path"], "src/main.py")

    def test_inline_test_definitions(self):
        text = """
Here's a test for the new function.

# File: tests/test_utils.py
def test_new_feature():
    assert True

Some more text.
"""
        result = parse_llm_output(text)
        self.assertEqual(len(result["patches"]), 0)
        self.assertEqual(len(result["unit_tests"]), 1)
        self.assertEqual(result["unit_tests"][0]["file_path"], "tests/test_utils.py")

    def test_no_patch_no_tests(self):
        text = "This is a response with no code."
        result = parse_llm_output(text)
        self.assertEqual(len(result["patches"]), 0)
        self.assertEqual(len(result["unit_tests"]), 0)

    def test_multiple_patches_and_tests(self):
        text = """
First, a patch for `file1.py`.
diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,1 +1,1 @@
-old_line
+new_line

And a test for it:
```python
def test_file1_change():
    pass
```

Second, a patch for `file2.py`.
diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,1 +1,1 @@
-another_old_line
+another_new_line

And an inline test for `file2.py`:
def test_file2_change():
    pass
"""
        result = parse_llm_output(text)
        self.assertEqual(len(result["patches"]), 2)
        self.assertEqual(len(result["unit_tests"]), 2)
        self.assertEqual(result["unit_tests"][0]["file_path"], "file1.py")
        self.assertEqual(result["unit_tests"][1]["file_path"], "file2.py")

    def test_missing_path(self):
        text = """
```python
def test_something():
    pass
```
"""
        result = parse_llm_output(text)
        self.assertEqual(len(result["unit_tests"]), 1)
        self.assertIsNone(result["unit_tests"][0]["file_path"])

    def test_windows_path(self):
        text = """
# File: src\\main.py
def test_windows_path():
    pass
"""
        result = parse_llm_output(text)
        self.assertEqual(len(result["unit_tests"]), 1)
        self.assertEqual(result["unit_tests"][0]["file_path"], "src/main.py")

if __name__ == '__main__':
    unittest.main()
