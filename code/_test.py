"""用法: python -I MAIN/code/_test.py —— 设好 FIN_ROOT 后跑 code/tests 下的全部单元测试"""
import os
import sys
import unittest
from pathlib import Path

code = Path(__file__).resolve().parent
os.environ["FIN_ROOT"] = str(code.parent)
sys.path.insert(0, str(code))

unittest.main(module=None, argv=["x", "discover", "-s", str(code / "tests"), "-p", "test_*.py"])
