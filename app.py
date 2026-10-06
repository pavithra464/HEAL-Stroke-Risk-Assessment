"""
HEAL Root Streamlit Entry Point
Enables seamless deployment on Streamlit Community Cloud whether the main file path
is configured as 'app.py' or 'HEAL_Streamlit/app.py'.
"""
import os
import sys

heal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HEAL_Streamlit")
sys.path.insert(0, heal_dir)

target_file = os.path.join(heal_dir, "app.py")
with open(target_file, "r") as f:
    code = compile(f.read(), target_file, 'exec')
    exec(code, {'__file__': target_file, '__name__': '__main__'})
