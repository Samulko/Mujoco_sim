import os
import sys

# Add the src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(src_dir)

from user_speak.main import main

if __name__ == "__main__":
    main()
