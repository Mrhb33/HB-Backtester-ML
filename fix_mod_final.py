import os

go_mod_path = 'c:/HB_bactest_checker/go.mod'

# Write clean go.mod
go_mod_content = '''module hb_bactest_checker

go 1.24.0

require golang.org/x/term v0.39.0
require (
	github.com/aymanbagabas/go-osc52/v2 v2.0.1
	github.com/charmbracelet/bubbles v0.21.0
	github.com/charmbracelet/bubbletea v1.3.10
	github.com/charmbracelet/colorprofile v0.2.3.20250311203215-f60798e515dc
	github.com/charmbracelet/harmonica v0.2.0
	github.com/charmbracelet/lipgloss v1.1.0
	github.com/charmbracelet/x/ansi v0.10.1
	github.com/charmbracelet/x/cellbuf v0.0.13.0.20250311204145-2c3ea96c31dd
	github.com/charmbracelet/x/term v0.2.1
	github.com/erikgeiser/coninput v0.0.0.20211004153227-1c3628e74d0f
	github.com/lucasb-eyer/go-colorful v1.2.0
	github.com/mattn/go-isatty v0.0.20
	github.com/mattn/go-localereader v0.0.1
	github.com/mattn/go-runewidth v0.0.16
	github.com/muesli/ansi v0.0.0.20230316100256-276c6243b2f6
	github.com/muesli/cancelreader v0.0.2
	github.com/muesli/termenv v0.0.16
	github.com/rivo/uniseg v0.4.7
	github.com/xo/terminfo v0.0.0.20220910002029-abceb7e1c41e // indirect
	github.com/gorilla/websocket v1.5.1
	golang.org/x/sys v0.40.0
	golang.org/x/text v0.3.8
)
'''

with open(go_mod_path, 'w', encoding='utf-8') as f:
    f.write(go_mod_content)

print("Successfully wrote clean go.mod")
