#!/usr/bin/env python3
"""
Test the working demo automatically
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from working_demo import run_working_demo

if __name__ == "__main__":
    print("🚀 Running automatic demo test...")
    try:
        run_working_demo()
    except Exception as e:
        print(f"Demo test failed: {e}")
        import traceback
        traceback.print_exc()