@echo off
cd /d c:\HB_bactest_checker
echo module hb_bactest_checker > go.mod
echo. >> go.mod
echo go 1.24.0 >> go.mod
echo. >> go.mod
echo require golang.org/x/term v0.39.0 >> go.mod
echo require ( >> go.mod
echo 	github.com/aymanbagabas/go-osc52/v2 v2.0.1 // indirect >> go.mod
echo 	github.com/charmbracelet/bubbles v0.21.0 // indirect >> go.mod
echo 	github.com/charmbracelet/bubbletea v1.3.10 // indirect >> go.mod
echo 	github.com/charmbracelet/colorprofile v0.2.3.0.20250311203215-f60798e515dc // indirect >> go.mod
echo 	github.com/charmbracelet/harmonica v0.2.0 // indirect >> go.mod
echo 	github.com/charmbracelet/lipgloss v1.1.0 // indirect >> go.mod
echo 	github.com/charmbracelet/x/ansi v0.10.1 // indirect >> go.mod
echo 	github.com/charmbracelet/x/cellbuf v0.0.13.0.20250311204145-2c3ea96c31dd // indirect >> go.mod
echo 	github.com/charmbracelet/x/term v0.2.1 // indirect >> go.mod
echo 	github.com/erikgeiser/coninput v0.0.0.20211004153227-1c3628e74d0f // indirect >> go.mod
echo 	github.com/lucasb-eyer/go-colorful v1.2.0 // indirect >> go.mod
echo 	github.com/mattn/go-isatty v0.0.20 // indirect >> go.mod
echo 	github.com/mattn/go-localereader v0.0.1 // indirect >> go.mod
echo 	github.com/mattn/go-runewidth v0.0.16 // indirect >> go.mod
echo 	github.com/muesli/ansi v0.0.0.20230316100256-276c6243b2f6 // indirect >> go.mod
echo 	github.com/muesli/cancelreader v0.2.2 // indirect >> go.mod
echo 	github.com/muesli/termenv v0.16.0 // indirect >> go.mod
echo 	github.com/rivo/uniseg v0.4.7 // indirect >> go.mod
echo 	github.com/xo/terminfo v0.0.0.20220910002029-abceb7e1c41e // indirect >> go.mod
echo 	github.com/gorilla/websocket v1.5.1 // For WebSocket support >> go.mod
echo 	golang.org/x/sys v0.40.0 // indirect >> go.mod
echo 	golang.org/x/text v0.3.8 // indirect >> go.mod
echo ) >> go.mod
echo Done!
