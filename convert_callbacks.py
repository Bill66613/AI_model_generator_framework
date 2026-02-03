"""Convert all callback files to use register_callbacks(app) pattern."""
import re
import os

callback_files = [
    'callbacks/preprocessing_callbacks.py',
    'callbacks/feature_engineering_callbacks.py',
    'callbacks/training_callbacks.py',
    'callbacks/code_generation_callbacks.py'
]

for filepath in callback_files:
    print(f"\n{'='*60}")
    print(f"Processing: {filepath}")
    print('='*60)
    
    if not os.path.exists(filepath):
        print(f"⚠️ File not found: {filepath}")
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already converted
    if 'def register_callbacks(app):' in content:
        print(f"✅ Already converted: {filepath}")
        continue
    
    # Step 1: Replace 'callback' with nothing in the import, but keep the comma structure
    # Match patterns like: "from dash import ..., callback, ..." 
    content = re.sub(r'(\bfrom\s+dash\s+import\s+[^;\n]*?),\s*callback\s*,', r'\1,', content)
    content = re.sub(r'(\bfrom\s+dash\s+import\s+[^;\n]*?),\s*callback\b', r'\1', content)
    content = re.sub(r'\bfrom\s+dash\s+import\s+callback\s*,', 'from dash import ', content)
    
    # Step 2: Find where to insert register_callbacks
    first_callback_match = re.search(r'^@callback', content, re.MULTILINE)
    
    if not first_callback_match:
        print(f"⚠️ No @callback decorators found in {filepath}")
        continue
    
    first_callback_pos = first_callback_match.start()
    
    # Split content
    before_callbacks = content[:first_callback_pos].rstrip()
    callbacks_section = content[first_callback_pos:]
    
    # Step 3: Build new content with register_callbacks wrapper
    new_content = before_callbacks + '\n\n\ndef register_callbacks(app):\n    """Register all callbacks with the app."""\n'
    
    # Step 4: Process each line in callbacks section
    lines = callbacks_section.split('\n')
    for line in lines:
        if line.lstrip().startswith('@callback'):
            # Replace @callback with @app.callback, keeping existing indentation + 4 spaces
            indent = len(line) - len(line.lstrip())
            new_content += ' ' * (indent + 4) + '@app.callback' + line.lstrip()[len('@callback'):] + '\n'
        elif line.lstrip().startswith('def ') and not line.startswith('    '):
            # Function definition - add 4 spaces
            new_content += '    ' + line + '\n'
        else:
            # All other content - add 4 spaces if non-empty
            if line.strip():
                new_content += '    ' + line + '\n'
            else:
                new_content += '\n'
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Converted: {filepath}")

print(f"\n{'='*60}")
print("Conversion complete! Verifying syntax...")
print('='*60)

# Verify all files
for filepath in callback_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            compile(f.read(), filepath, 'exec')
        print(f"✅ {filepath} - syntax valid")
    except SyntaxError as e:
        print(f"❌ {filepath} - syntax error at line {e.lineno}: {e.msg}")

