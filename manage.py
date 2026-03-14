#!/usr/bin/env python
"""
Climora Management Script

This script provides a unified command-line interface for common developer
tasks within the Climora project.

Usage:
    python manage.py runserver      - Starts the Flask development server on port 5000.
    python manage.py test           - Runs the entire unittest suite inside `tests/`.
    python manage.py generate-data  - Invokes the synthetic data generator.
    python manage.py evaluate-model - Runs the offline K-Means evaluation pipeline.
"""

import sys
import argparse
import unittest
import os
import subprocess

def run_tests():
    """Discover and run all unittests in the tests directory."""
    print("Running Climora Test Suite...")
    loader = unittest.TestLoader()
    tests = loader.discover('tests')
    testRunner = unittest.runner.TextTestRunner(verbosity=2)
    testRunner.run(tests)

def run_server():
    """Start the Flask application."""
    print("Starting Climora Development Server...")
    from app import app
    app.run(debug=True, port=5000, host='0.0.0.0')

def run_data_generator():
    """Run the synthetic data generation script."""
    print("Running Data Generator...")
    # Using subprocess to execute the script in its own context
    script_path = os.path.join('scripts', 'data_generator.py')
    if os.path.exists(script_path):
        subprocess.run([sys.executable, script_path])
    else:
        print(f"Error: Could not find {script_path}")

def run_model_evaluator():
    """Run the offline ML model evaluation script."""
    print("Running Model Evaluator...")
    script_path = os.path.join('scripts', 'model_pipeline.py')
    if os.path.exists(script_path):
        subprocess.run([sys.executable, script_path])
    else:
        print(f"Error: Could not find {script_path}")

def main():
    """Parse CLI arguments and execute the correct management command."""
    parser = argparse.ArgumentParser(description="Climora Management Utility")
    parser.add_argument('command', help="runserver | test | generate-data | evaluate-model")
    
    args = parser.parse_args()
    
    if args.command == 'runserver':
        run_server()
    elif args.command == 'test':
        run_tests()
    elif args.command == 'generate-data':
        run_data_generator()
    elif args.command == 'evaluate-model':
        run_model_evaluator()
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
