package main

import (
	"fmt"
	"log"
	"net"
	"net/http"
	"sync"
	"time"

	"hb_bactest_checker/logx"

	"github.com/gorilla/websocket"
)

// WSHub manages WebSocket connections and broadcasts
type WSHub struct {
	clients    map[*websocket.Conn]bool
	broadcast  chan WSMessage
	register   chan *websocket.Conn
	unregister chan *websocket.Conn
	mutex      sync.RWMutex
	running    bool
}

// WSMessage represents a WebSocket message
type WSMessage struct {
	Type string      `json:"type"` // "progress", "metrics", "trade_log", "oos_stats", "elite_log", "status", etc.
	Data interface{} `json:"data"` // Payload data
	Time int64       `json:"time"` // Unix timestamp
}

var wsHub *WSHub
var webDashboardEnabled = false

// WSMessageType constants
const (
	MsgTypeProgress       = "progress"
	MsgTypeMetrics        = "metrics"
	MsgTypeTradeLog       = "trade_log"
	MsgTypeOOSStats       = "oos_stats"
	MsgTypeMonthlyReturns = "monthly_returns"
	MsgTypeIndicatorData  = "indicator_data"
	MsgTypeProofData      = "proof_data"
	MsgTypeEliteLog       = "elite_log"
	MsgTypeHallOfFame     = "hall_of_fame"
	MsgTypeFoldResults    = "fold_results"
	MsgTypeExitReasons    = "exit_reasons"
	MsgTypeStatus         = "status"
	MsgTypeSignalEvent    = "signal_event"
	MsgTypeError          = "error"
	MsgTypeWarning        = "warning"
)

// InitWebServer initializes the WebSocket hub
func InitWebServer() {
	wsHub = &WSHub{
		clients:    make(map[*websocket.Conn]bool),
		broadcast:  make(chan WSMessage, 256),
		register:   make(chan *websocket.Conn),
		unregister: make(chan *websocket.Conn),
		running:    true,
	}
	go wsHub.run()
}

// StartWebServer starts the HTTP/WebSocket server
func StartWebServer(port int) error {
	InitWebServer()

	// Serve static files (dashboard.html)
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "dashboard.html")
	})

	// WebSocket endpoint
	http.HandleFunc("/ws", wsHub.handleWebSocket)

	// CORS middleware wrapper
	handler := corsMiddleware(http.DefaultServeMux)

	addr := fmt.Sprintf(":%d", port)
	fmt.Printf("\n%s Dashboard running at http://localhost%s\n", logx.Icon("info"), addr)
	fmt.Printf("%s Open this URL in your browser to view the dashboard\n", logx.Icon("info"))
	fmt.Printf("%s Press Ctrl+C to stop\n", logx.Icon("info"))

	return http.ListenAndServe(addr, handler)
}

// handleWebSocket handles WebSocket connections
func (hub *WSHub) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	// Upgrade HTTP to WebSocket
	ws, err := websocket.Upgrade(w, r, nil, 0, 0)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}

	// Register client
	hub.register <- ws
	defer func() {
		hub.unregister <- ws
		ws.Close()
	}()

	// Send buffered messages for new connections
	hub.sendBufferedMessages(ws)

	// Read messages from client
	for {
		var msg WSMessage
		err := ws.ReadJSON(&msg)
		if err != nil {
			break
		}
		// Client can send ping/heartbeat if needed
	}
}

// run processes messages in the hub
func (hub *WSHub) run() {
	for {
		select {
		case client := <-hub.register:
			hub.mutex.Lock()
			hub.clients[client] = true
			hub.mutex.Unlock()

		case client := <-hub.unregister:
			hub.mutex.Lock()
			if _, ok := hub.clients[client]; ok {
				delete(hub.clients, client)
			}
			hub.mutex.Unlock()

		case message := <-hub.broadcast:
			hub.mutex.RLock()
			for client := range hub.clients {
				err := client.WriteJSON(message)
				if err != nil {
					// Client disconnected, will be cleaned up by unregister
					continue
				}
			}
			hub.mutex.RUnlock()
		}
	}
}

// Broadcast sends a message to all connected clients
func Broadcast(msgType string, data interface{}) {
	if !webDashboardEnabled || wsHub == nil {
		return
	}

	msg := WSMessage{
		Type: msgType,
		Data: data,
		Time: time.Now().Unix(),
	}

	select {
	case wsHub.broadcast <- msg:
		// Message queued
	default:
		// Channel full, skip this message (backpressure protection)
	}
}

// sendBufferedMessages sends recent history to new connections
func (hub *WSHub) sendBufferedMessages(ws *websocket.Conn) {
	// Send current status
	statusMsg := WSMessage{
		Type: MsgTypeStatus,
		Data: map[string]interface{}{
			"status": "running",
			"msg":    "Dashboard connected",
		},
		Time: time.Now().Unix(),
	}
	ws.WriteJSON(statusMsg)
}

// corsMiddleware adds CORS headers to responses
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// FindAvailablePort finds an available port starting from startPort
func FindAvailablePort(startPort int) int {
	for port := startPort; port < 9000; port++ {
		ln, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
		if err == nil {
			ln.Close()
			return port
		}
	}
	return startPort // fallback
}

// Message structures for WebSocket payloads

// ProgressData represents generation progress
type ProgressData struct {
	Generated       int64   `json:"generated"`
	Tested          int64   `json:"tested"`
	Elites          int     `json:"elites"`
	RejectedSur     int64   `json:"rejected_surrogate"`
	RejectedSeen    int64   `json:"rejected_seen"`
	RejectedNovelty int64   `json:"rejected_novelty"`
	TimeElapsed     string  `json:"time_elapsed"`
	StratPerSec     float64 `json:"strat_per_sec"`
}

// MetricsData represents backtest metrics
type MetricsData struct {
	Score         float32 `json:"score"`
	Return        float32 `json:"return"`
	MaxDD         float32 `json:"max_dd"`
	WinRate       float32 `json:"win_rate"`
	Expectancy    float32 `json:"expectancy"`
	ProfitFactor  float32 `json:"profit_factor"`
	Trades        int     `json:"trades"`
	TotalHoldBars int     `json:"total_hold_bars"`
}

// TradeLogData represents a trade log entry
type TradeLogData struct {
	TradeID     int     `json:"trade_id"`
	SignalTime  string  `json:"signal_time"`
	EntryTime   string  `json:"entry_time"`
	ExitTime    string  `json:"exit_time"`
	Direction   string  `json:"direction"`
	EntryPrice  float32 `json:"entry_price"`
	ExitPrice   float32 `json:"exit_price"`
	PnL         float32 `json:"pnl"`
	HoldBars    int     `json:"hold_bars"`
	ExitReason  string  `json:"exit_reason"`
	StopPrice   float32 `json:"stop_price"`
	TPPrice     float32 `json:"tp_price"`
	TrailActive bool    `json:"trail_active"`
}

// OOSStatsData represents out-of-sample statistics
type OOSStatsData struct {
	GeoAvgMonthly       float64 `json:"geo_avg_monthly"`
	ActiveGeoAvgMonthly float64 `json:"active_geo_avg_monthly"`
	MedianMonthly       float64 `json:"median_monthly"`
	MedianAllMonths     float64 `json:"median_all_months"`
	MinMonth            float64 `json:"min_month"`
	StdMonth            float64 `json:"std_month"`
	MaxDD               float64 `json:"max_dd"`
	TotalMonths         int     `json:"total_months"`
	TotalTrades         int     `json:"total_trades"`
	MinTradesPerMonth   int     `json:"min_trades_per_month"`
	ActiveMonthsCount   int     `json:"active_months_count"`
	ActiveMonthsRatio   float64 `json:"active_months_ratio"`
	SparseMonthsCount   int     `json:"sparse_months_count"`
	SparseMonthsRatio   float64 `json:"sparse_months_ratio"`
	OOSProfitFactor     float64 `json:"oos_profit_factor"`
	OOSExpectancy       float64 `json:"oos_expectancy"`
	OOSWinRate          float64 `json:"oos_win_rate"`
	Rejected            bool    `json:"rejected"`
	RejectReason        string  `json:"reject_reason"`
	RejectCode          int     `json:"reject_code"`
}

// MonthlyReturnData represents a single month's return
type MonthlyReturnData struct {
	Month  int     `json:"month"`
	Return float64 `json:"return"`
	DD     float64 `json:"dd"`
	Trades int     `json:"trades"`
}

// SignalEventData represents a signal event
type SignalEventData struct {
	BarIndex      int     `json:"bar_index"`
	Time          string  `json:"time"`
	ClosePrice    float64 `json:"close_price"`
	RegimeOK      bool    `json:"regime_ok"`
	EntryOK       bool    `json:"entry_ok"`
	SubConditions string  `json:"sub_conditions"`
}

// EliteLogData represents an elite strategy
type EliteLogData struct {
	Rank       int     `json:"rank"`
	Score      float32 `json:"score"`
	ValScore   float32 `json:"val_score"`
	Return     float32 `json:"return"`
	MaxDD      float32 `json:"max_dd"`
	WinRate    float32 `json:"win_rate"`
	Trades     int     `json:"trades"`
	IsPreElite bool    `json:"is_pre_elite"`
	Timestamp  string  `json:"timestamp"`
}

// HallOfFameData represents Hall of Fame state
type HallOfFameData struct {
	K      int            `json:"k"`
	Elites []EliteLogData `json:"elites"`
}

// Helper functions for sending specific message types

func SendProgress(generated, tested, rejectedSur, rejectedSeen, rejectedNovelty int64, elites int, timeElapsed string, stratPerSec float64) {
	data := ProgressData{
		Generated:       generated,
		Tested:          tested,
		Elites:          elites,
		RejectedSur:     rejectedSur,
		RejectedSeen:    rejectedSeen,
		RejectedNovelty: rejectedNovelty,
		TimeElapsed:     timeElapsed,
		StratPerSec:     stratPerSec,
	}
	Broadcast(MsgTypeProgress, data)
}

func SendMetrics(score, ret, maxDD, winRate, expectancy, profitFactor float32, trades, totalHoldBars int) {
	data := MetricsData{
		Score:         score,
		Return:        ret,
		MaxDD:         maxDD,
		WinRate:       winRate,
		Expectancy:    expectancy,
		ProfitFactor:  profitFactor,
		Trades:        trades,
		TotalHoldBars: totalHoldBars,
	}
	Broadcast(MsgTypeMetrics, data)
}

func SendTradeLog(trade Trade) {
	data := TradeLogData{
		TradeID:    len(trade.Proofs) + 1, // Approximate trade ID
		SignalTime: trade.SignalTime.Format("2006-01-02 15:04:05"),
		EntryTime:  trade.EntryTime.Format("2006-01-02 15:04:05"),
		ExitTime:   trade.ExitTime.Format("2006-01-02 15:04:05"),
		Direction: func() string {
			if trade.Direction == 1 {
				return "LONG"
			} else {
				return "SHORT"
			}
		}(),
		EntryPrice:  trade.EntryPrice,
		ExitPrice:   trade.ExitPrice,
		PnL:         trade.PnL,
		HoldBars:    trade.HoldBars,
		ExitReason:  trade.Reason,
		StopPrice:   trade.StopPrice,
		TPPrice:     trade.TPPrice,
		TrailActive: trade.TrailActive,
	}
	Broadcast(MsgTypeTradeLog, data)
}

func SendOOSStats(stats OOSStats) {
	data := OOSStatsData{
		GeoAvgMonthly:       stats.GeoAvgMonthly,
		ActiveGeoAvgMonthly: stats.ActiveGeoAvgMonthly,
		MedianMonthly:       stats.MedianMonthly,
		MedianAllMonths:     stats.MedianAllMonths,
		MinMonth:            stats.MinMonth,
		StdMonth:            stats.StdMonth,
		MaxDD:               stats.MaxDD,
		TotalMonths:         stats.TotalMonths,
		TotalTrades:         stats.TotalTrades,
		MinTradesPerMonth:   stats.MinTradesPerMonth,
		ActiveMonthsCount:   stats.ActiveMonthsCount,
		ActiveMonthsRatio:   stats.ActiveMonthsRatio,
		SparseMonthsCount:   stats.SparseMonthsCount,
		SparseMonthsRatio:   stats.SparseMonthsRatio,
		OOSProfitFactor:     stats.OOSProfitFactor,
		OOSExpectancy:       stats.OOSExpectancy,
		OOSWinRate:          stats.OOSWinRate,
		Rejected:            stats.Rejected,
		RejectReason:        stats.RejectReason,
		RejectCode:          int(stats.RejectCode),
	}
	Broadcast(MsgTypeOOSStats, data)
}

func SendMonthlyReturns(monthlyReturns []MonthlyReturn) {
	data := make([]MonthlyReturnData, len(monthlyReturns))
	for i, mr := range monthlyReturns {
		data[i] = MonthlyReturnData{
			Month:  mr.Month,
			Return: mr.Return,
			DD:     mr.DD,
			Trades: mr.Trades,
		}
	}
	Broadcast(MsgTypeMonthlyReturns, data)
}

func SendStatus(status, msg string) {
	data := map[string]interface{}{
		"status": status,
		"msg":    msg,
	}
	Broadcast(MsgTypeStatus, data)
}

func SendError(msg string) {
	Broadcast(MsgTypeError, msg)
}

func SendWarning(msg string) {
	Broadcast(MsgTypeWarning, msg)
}
