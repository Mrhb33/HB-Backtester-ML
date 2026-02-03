import re

# Read go.mod
with open('go.mod', 'r', newline='\n') as f:
    content = f.read()

# Find the x/o/terminfo line and add websocket after it
pattern = r'(github\.com/xo/terminfo v0\.0\.0\.20220910002029-abceb7e1c41e // indirect)\r\nreplacement = r'\1\n\ngithub.com/gorilla/websocket v1.5.1\n\n'

new_content = re.sub(pattern, replacement, content)

# Write back
with open('go.mod', 'w', newline='\n') as f:
    f.write(new_content)

print("Successfully added gorilla/websocket to go.mod")
