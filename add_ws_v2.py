# Read go.mod
with open('go.mod', 'r', newline='') as f:
    lines = f.readlines()

# Find the line with x/o/terminfo and add websocket after it
new_lines = []
for line in lines:
    new_lines.append(line)
    if 'github.com/x/o/terminfo' in line and 'gorilla/websocket' not in lines:
        # Add websocket line after this one
        new_lines[-1] = line + '\n\tgithub.com/gorilla/websocket v1.5.1'
        print("Added websocket dependency line")
        break

# Write back
with open('go.mod', 'w', newline='\n') as f:
    f.writelines(new_lines)

print("Successfully updated go.mod")
