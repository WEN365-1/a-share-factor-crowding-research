"""用法: python -I MAIN/code/_run.py 09_regressions.py [参数...]  —— 设置 FIN_ROOT 后运行指定脚本"""
import os, runpy, sys
from pathlib import Path
code = Path(__file__).resolve().parent
os.environ["FIN_ROOT"] = str(code.parent)
script = sys.argv[1]
sys.argv = [str(code / script)] + sys.argv[2:]
sys.path.insert(0, str(code))
runpy.run_path(str(code / script), run_name="__main__")
