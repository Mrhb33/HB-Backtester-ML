@echo off
cd /d c:\HB_bactest_checker
go mod edit -require=github.com/gorilla/websocket@latest -dropreplace=github.com/gorilla/websocket
go mod tidy
echo Dependencies updated successfully
