import os
import re

go_mod_path = 'c:/HB_bactest_checker/go.mod'

# Read go.mod
with open(go_mod_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Process lines
new_lines = []
for i, line in enumerate(lines):
    # Skip lines that have version timestamps
    if re.search(r'version "v\d+\.\d+', line):
        print(f"Skipping versioned line {i+1}")
        continue
    
    # Add gorilla/websocket line after x/o/terminfo line
    new_lines.append(line)
    if 'github.com/x/o/terminfo' in line:
        # Insert gorilla/websocket dependency on the next line
        new_lines.append('\tgithub.com/gorilla/websocket v1.5.1')
        print(f"Inserted gorilla/websocket at line {i+2}")

# Write back
with open(go_mod_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))

print("Successfully updated go.mod")
