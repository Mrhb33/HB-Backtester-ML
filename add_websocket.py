import os

go_mod_path = 'c:/HB_bactest_checker/go.mod'

with open(go_mod_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skipped_old_net = False

for i, line in enumerate(lines):
    # Remove old golang.org/x/net line
    if 'golang.org/x/net v0.32.0 // indirect' in line:
        print(f"Skipping old golang.org/x/net line at line {i+1}")
        skipped_old_net = True
        continue

    # Insert gorilla/websocket line after x/o/terminfo
    if 'github.com/x/o/terminfo v0.0.0.20220910002029-abceb7e1c41e // indirect' in line and 'gorilla' not in lines[i+1:]:
        new_lines.append('\tgithub.com/gorilla/websocket v1.5.1')
        print(f"Inserted gorilla/websocket at line {i+1}")
        new_lines.append(line)
        continue

    # Skip next line (the old terminfo line we want to remove)
    if i + 1 < len(lines) and 'github.com/xo/terminfo' in lines[i+1]:
        print(f"Skipping old terminfo line at line {i+2}")
        continue

    new_lines.append(line)

with open(go_mod_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Successfully updated go.mod")
