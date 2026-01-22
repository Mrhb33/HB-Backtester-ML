package logx

import (
	"fmt"
	"io"
	"os"
	"time"
	"text/tabwriter"
)

// PrintFeatureRow - prints feature values as aligned table
func PrintFeatureRow(prefix string, ts time.Time, bar int, m map[string]float64) {
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(w, "%s\tbar=%d\ttime=%s\n", prefix, bar, ts.UTC().Format(time.RFC3339))
	keys := []string{"EMA10","EMA20","EMA200","MACD","MACD_Signal","BB_Width50","MinusDI","VolSMA20","Imbalance","SwingHigh","SwingLow"}
	for _, k := range keys {
		if v, ok := m[k]; ok {
			fmt.Fprintf(w, "  %s:\t%.4f\n", k, v)
		}
	}
	w.Flush()
}

// NewTableWriter creates a tabwriter for custom output
func NewTableWriter(w io.Writer) *tabwriter.Writer {
	return tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
}

// PrintTradeWindowHeader - prints trade window header
func PrintTradeWindowHeader() {
	fmt.Printf("\n%s  %s  SINGLE TRADE MODE: TRADE WINDOW DUMP\n",
		C(gray, time.Now().UTC().Format("15:04:05Z")),
		Channel("SIG "),
	)
}

// PrintTradeWindowBar - prints a single bar in trade window
func PrintTradeWindowBar(i int, isBeforeEntry, isEntry, isExit, isAfterExit bool) {
	fmt.Printf("\n--- Bar i=%d ", i)
	if isBeforeEntry {
		fmt.Printf("(BEFORE ENTRY) ---\n")
	} else if isEntry {
		fmt.Printf("(ENTRY BAR) ---\n")
	} else if isExit {
		fmt.Printf("(EXIT BAR) ---\n")
	} else if isAfterExit {
		fmt.Printf("(AFTER EXIT) ---\n")
	} else {
		fmt.Printf("---\n")
	}
}

// PrintTradeWindowOHLC - prints OHLC for a bar
func PrintTradeWindowOHLC(open, high, low, close float32) {
	fmt.Printf("  OHLC: O=%.2f H=%.2f L=%.2f C=%.2f\n", open, high, low, close)
}

// PrintTradeWindowFeatures - prints feature values at a bar
func PrintTradeWindowFeatures(features map[string]float32) {
	fmt.Printf("  Features:\n")
	for name, value := range features {
		fmt.Printf("    %s=%.4f ", name, value)
	}
	fmt.Printf("\n")
}
