# Read go.mod, convert line endings, append websocket, convert back
import os

# Read with Windows line endings
with open('go.mod', 'r', newline='\r\n') as f:
    content = f.read()

# Convert to Unix line endings
lines = content.replace('\r\n', '\n').split('\n')

# Find where to insert websocket dependency (before the closing parenthesis)
insert_idx = -1
for i, line in enumerate(lines):
    if line.strip() == ')':
        insert_idx = i - 1
        break

if insert_idx > 0:
    # Insert websocket line
    lines.insert(insert_idx, '\tgithub.com/gorilla/websocket v1.5.1')
    print(f"Inserted websocket at line {insert_idx + 1}")
else:
    print("Could not find insertion point")

# Write back with Unix line endings
with open('go.mod', 'w', newline='\n') as f:
    f.write('\n'.join(lines))

print("Successfully updated go.mod with Unix line endings")
