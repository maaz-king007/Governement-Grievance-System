#!/usr/bin/env python3
"""
setup_and_run.py
One-click setup: generates data, trains models, starts the web server.
"""
import os
import sys
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)


def step(msg):
    print(f"\n{'='*55}")
    print(f"  {msg}")
    print('='*55)


def main():
    step("Step 1/3 — Generating Dataset")
    from data.generate_dataset import generate
    generate()

    step("Step 2/3 — Training Models")
    import train_models
    train_models.main()

    step("Step 3/3 — Starting Web Server")
    from app import app, init_db
    init_db()
    print("\n🚀 Server running at: http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    app.run(debug=False, port=5000, host="0.0.0.0")


if __name__ == "__main__":
    main()
