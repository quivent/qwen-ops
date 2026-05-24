package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "qwen-ops",
	Short: "Qwen MTP optimization toolkit",
	Long: `qwen-ops applies Qwen Multi-Token Prediction optimizations
discovered in the quivent research repos. It manages llama.cpp
and vLLM patches, launches optimized inference, and benchmarks
MTP performance on GH200 hardware.`,
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func init() {
	rootCmd.AddCommand(serveCmd)
	rootCmd.AddCommand(mtpCmd)
	rootCmd.AddCommand(patchesCmd)
	rootCmd.AddCommand(statusCmd)
	rootCmd.AddCommand(downloadCmd)
}
