#!/usr/bin/env python3
"""
HAR Framework Project Verification Script
Validates project structure, dependencies, and functionality
"""

import os
import sys
import importlib.util
import subprocess
from pathlib import Path
import json

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return os.path.exists(file_path)

def check_directory_exists(dir_path: str) -> bool:
    """Check if a directory exists."""
    return os.path.isdir(dir_path)

def check_python_module(module_name: str) -> bool:
    """Check if a Python module can be imported."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except ImportError:
        return False

def verify_project_structure():
    """Verify the project directory structure."""
    print("🏗️  Verifying Project Structure...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        'TODO.md',
        'PROJECT_SUMMARY.md'
    ]
    
    required_directories = [
        'assets',
        'callbacks',
        'config',
        'data',
        'deployment',
        'docs',
        'layouts',
        'models',
        'persistent_data',
        'tests',
        'utils'
    ]
    
    # Check files
    missing_files = []
    for file in required_files:
        if not check_file_exists(file):
            missing_files.append(file)
    
    # Check directories
    missing_dirs = []
    for directory in required_directories:
        if not check_directory_exists(directory):
            missing_dirs.append(directory)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    if missing_dirs:
        print(f"❌ Missing directories: {', '.join(missing_dirs)}")
        return False
    
    print("✅ Project structure is complete!")
    return True

def verify_key_modules():
    """Verify that key Python modules exist and can be imported."""
    print("\n📦 Verifying Key Modules...")
    
    key_modules = [
        'callbacks.data_callbacks',
        'callbacks.preprocessing_callbacks', 
        'callbacks.training_callbacks',
        'config.config',
        'layouts.data_upload',
        'layouts.preprocessing',
        'layouts.training',
        'utils.data_processing',
        'utils.model_training',
        'deployment.code_generator'
    ]
    
    failed_imports = []
    for module in key_modules:
        try:
            # Convert relative imports to file paths for verification
            module_path = module.replace('.', os.sep) + '.py'
            if not check_file_exists(module_path):
                failed_imports.append(f"{module} (file not found)")
        except Exception as e:
            failed_imports.append(f"{module} ({str(e)})")
    
    if failed_imports:
        print(f"❌ Failed module checks: {', '.join(failed_imports)}")
        return False
    
    print("✅ All key modules are present!")
    return True

def verify_dependencies():
    """Verify that required dependencies are listed in requirements.txt."""
    print("\n📋 Verifying Dependencies...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        critical_packages = [
            'dash', 'plotly', 'pandas', 'numpy', 'scikit-learn',
            'scipy', 'tensorflow', 'pytest', 'seaborn', 'matplotlib'
        ]
        
        missing_packages = []
        for package in critical_packages:
            if package not in requirements.lower():
                missing_packages.append(package)
        
        if missing_packages:
            print(f"⚠️  Potentially missing packages: {', '.join(missing_packages)}")
        else:
            print("✅ All critical dependencies are listed!")
        
        return len(missing_packages) == 0
        
    except FileNotFoundError:
        print("❌ requirements.txt not found!")
        return False

def verify_data_structure():
    """Verify data directory structure."""
    print("\n💾 Verifying Data Structure...")
    
    data_dirs = ['data/100hz', 'data/100hz_mod', 'data/100hz_smoothed']
    persistent_data_exists = check_directory_exists('persistent_data')
    
    data_structure_ok = True
    for data_dir in data_dirs:
        if not check_directory_exists(data_dir):
            print(f"⚠️  Data directory missing: {data_dir}")
            data_structure_ok = False
    
    if not persistent_data_exists:
        print("⚠️  Persistent data directory missing")
        data_structure_ok = False
    
    if data_structure_ok:
        print("✅ Data structure is properly organized!")
    
    return data_structure_ok

def verify_tests():
    """Verify test structure."""
    print("\n🧪 Verifying Test Structure...")
    
    test_files = ['tests/test_main.py']
    
    missing_tests = []
    for test_file in test_files:
        if not check_file_exists(test_file):
            missing_tests.append(test_file)
    
    if missing_tests:
        print(f"❌ Missing test files: {', '.join(missing_tests)}")
        return False
    
    print("✅ Test structure is complete!")
    return True

def verify_documentation():
    """Verify documentation structure."""
    print("\n📚 Verifying Documentation...")
    
    doc_files = ['docs/README.md', 'PROJECT_SUMMARY.md']
    
    missing_docs = []
    for doc_file in doc_files:
        if not check_file_exists(doc_file):
            missing_docs.append(doc_file)
    
    if missing_docs:
        print(f"⚠️  Missing documentation: {', '.join(missing_docs)}")
        return False
    
    print("✅ Documentation structure is complete!")
    return True

def generate_verification_report():
    """Generate a comprehensive verification report."""
    print("\n📊 Generating Verification Report...")
    
    report = {
        "project_structure": verify_project_structure(),
        "key_modules": verify_key_modules(), 
        "dependencies": verify_dependencies(),
        "data_structure": verify_data_structure(),
        "tests": verify_tests(),
        "documentation": verify_documentation()
    }
    
    # Count successful verifications
    passed = sum(report.values())
    total = len(report)
    
    print(f"\n📈 Verification Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All verifications passed! Project is ready for use.")
        return True
    else:
        print("⚠️  Some verifications failed. Please review the issues above.")
        return False

def print_next_steps():
    """Print recommended next steps for the user."""
    print("\n🚀 Recommended Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the application: python app.py")
    print("3. Access the web interface: http://localhost:8050")
    print("4. Run tests: pytest tests/ -v")
    print("5. Read documentation: docs/README.md")
    print("6. Check project summary: PROJECT_SUMMARY.md")

def main():
    """Main verification function."""
    print("🔍 HAR Framework Project Verification")
    print("=" * 50)
    
    # Change to the script's directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Run all verifications
    success = generate_verification_report()
    
    # Print next steps regardless of verification results
    print_next_steps()
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
