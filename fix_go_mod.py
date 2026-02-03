import os
import re

go_mod_path = 'c:/HB_bactest_checker/go.mod'

with open(go_mod_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix version strings by removing timestamp suffix
# Pattern: version "v0.2.3.0.20250311203215-f60798e515dc" -> version "v0.2.3"
lines = content.split('\n')
new_lines = []

for i, line in enumerate(lines):
    # Look for version strings with timestamp suffix
    match = re.match(r'version "v(\d+\.\d+\.\d+\.\d+)\.(\d{14}-f\d{6}-\w{4}f\d{8}e\d{5}dc)"', line)
    if match:
        # Extract just the semantic version
        version = match.group(1)
        # Replace the full version string
        new_line = line.replace(f'version "{version}"', f'version "{version}"')
        new_lines.append(new_line)
        print(f"Fixed version string at line {i+1}: {version}")
    else:
        new_lines.append(line)

with open(go_mod_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))

print("Successfully fixed all version strings in go.mod")
