#!/usr/bin/env python3
"""
Test runner for Claude Ollama Proxy.
Run all tests or specific test modules.
"""

import unittest
import sys
import os
import argparse

def main():
    """Run tests based on command-line arguments."""
    parser = argparse.ArgumentParser(description='Run tests for Claude Ollama Proxy')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--api', action='store_true', help='Run API tests only')
    parser.add_argument('--dashboard', action='store_true', help='Run dashboard tests only')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # If no specific test type is selected, run all tests
    if not (args.unit or args.api or args.dashboard or args.all):
        args.all = True
    
    # Set up the test loader
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests based on command-line arguments
    if args.unit or args.all:
        unit_tests = loader.discover(os.path.dirname(__file__), pattern='test_metrics.py')
        suite.addTest(unit_tests)
    
    if args.api or args.all:
        api_tests = loader.discover(os.path.dirname(__file__), pattern='test_api.py')
        suite.addTest(api_tests)
    
    if args.dashboard or args.all:
        dashboard_tests = loader.discover(os.path.dirname(__file__), pattern='test_dashboard.py')
        suite.addTest(dashboard_tests)
    
    # Set up the test runner
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    # Run the tests
    result = runner.run(suite)
    
    # Return non-zero exit code if any tests failed
    sys.exit(not result.wasSuccessful())

if __name__ == '__main__':
    main()