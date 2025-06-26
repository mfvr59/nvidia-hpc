#!/usr/bin/env python3
"""
Check for potential linter issues in the services
"""

import ast
import sys
from pathlib import Path

def check_file(file_path):
    """Check a Python file for potential issues"""
    print(f"\n🔍 Checking {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content)
        
        # Check for imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        print(f"  📦 Imports found: {len(imports)}")
        for imp in imports:
            print(f"    - {imp}")
        
        # Check for try/except blocks
        try_blocks = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                try_blocks += 1
        
        print(f"  🔄 Try/except blocks: {try_blocks}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Check all service files"""
    services_dir = Path("backend/services")
    
    if not services_dir.exists():
        print("❌ Services directory not found")
        return
    
    service_files = list(services_dir.glob("*.py"))
    service_files = [f for f in service_files if f.name != "__init__.py"]
    
    print(f"🔍 Checking {len(service_files)} service files...")
    
    all_ok = True
    for service_file in service_files:
        if not check_file(service_file):
            all_ok = False
    
    if all_ok:
        print("\n✅ All service files are syntactically correct!")
        print("\n💡 If you're still seeing 'red' in your IDE, it might be:")
        print("   - Linter warnings about optional imports")
        print("   - Type checking issues with external packages")
        print("   - Missing type stubs for some packages")
        print("   - IDE-specific linting rules")
    else:
        print("\n❌ Some files have issues")

if __name__ == "__main__":
    main() 