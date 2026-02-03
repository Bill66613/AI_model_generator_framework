import re

with open('callbacks/data_callbacks.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

output_lines = []
inside_register = False
inside_callback_func = False
base_indent = ''

i = 0
while i < len(lines):
    line = lines[i]
    
    # Check if we're entering register_callbacks
    if 'def register_callbacks(app):' in line:
        inside_register = True
        output_lines.append(line)
        i += 1
        continue
    
    # If we're not inside register_callbacks yet, keep lines as-is
    if not inside_register:
        output_lines.append(line)
        i += 1
        continue
    
    # Inside register_callbacks
    stripped = line.lstrip()
    
    # Check for callback decorator
    if stripped.startswith('@app.callback('):
        # Callback decorators should be indented 4 spaces
        output_lines.append('    ' + stripped)
        i += 1
        continue
    
    # Check for function definition
    if stripped.startswith('def '):
        # Function definitions should be indented 4 spaces
        output_lines.append('    ' + stripped)
        inside_callback_func = True
        i += 1
        continue
    
    # If we're inside a callback function, add 4 more spaces
    if inside_callback_func:
        # Check if this is a new callback (end of previous function)
        if stripped.startswith('@app.callback(') or stripped.startswith('def register_callbacks'):
            inside_callback_func = False
            i -= 1  # Reprocess this line
            i += 1
            continue
        
        # Regular line inside function - add 4 spaces to existing indentation
        if line.strip():  # Not an empty line
            # Count existing indent
            existing_indent = len(line) - len(line.lstrip())
            # Add 4 more spaces
            output_lines.append('    ' + line)
        else:
            output_lines.append(line)  # Keep empty lines as-is
    else:
        output_lines.append(line)
    
    i += 1

with open('callbacks/data_callbacks.py', 'w', encoding='utf-8') as f:
    f.writelines(output_lines)

print("Indentation fixed successfully!")
