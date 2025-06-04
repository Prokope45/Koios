"""__main__.py

If `src` is ran as a module in the terminal, this will pass the CL args into
the main program class and method.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from src.main import Main
import sys

Main.main(sys.argv)
