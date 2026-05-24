package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

var (
	downloadDir   string
	downloadQuant string
	downloadRepo  string
)

var downloadCmd = &cobra.Command{
	Use:   "download",
	Short: "Download Qwen3.6-27B Q4_K_M GGUF",
	Long: `Download the Qwen3.6-27B quantized GGUF model from HuggingFace.

Uses huggingface-cli to download from unsloth/Qwen3.6-27B-GGUF.
Default quantization: Q4_K_M (best quality/speed for GH200).`,
	RunE: runDownload,
}

func init() {
	home, _ := os.UserHomeDir()

	downloadCmd.Flags().StringVarP(&downloadDir, "dir", "d", filepath.Join(home, "models"), "Download directory")
	downloadCmd.Flags().StringVarP(&downloadQuant, "quant", "q", "Q4_K_M", "Quantization level (Q4_K_M, Q5_K_M, Q8_0, etc.)")
	downloadCmd.Flags().StringVar(&downloadRepo, "repo", "unsloth/Qwen3.6-27B-GGUF", "HuggingFace repo")
}

func runDownload(cmd *cobra.Command, args []string) error {
	// Check for huggingface-cli
	hfBin, err := exec.LookPath("huggingface-cli")
	if err != nil {
		// Also check for hf
		hfBin, err = exec.LookPath("hf")
		if err != nil {
			return fmt.Errorf("huggingface-cli not found; install with: pip install huggingface-hub")
		}
	}

	// Ensure download directory exists
	if err := os.MkdirAll(downloadDir, 0755); err != nil {
		return fmt.Errorf("creating download dir: %w", err)
	}

	pattern := fmt.Sprintf("*%s*", downloadQuant)

	fmt.Printf("=== qwen-ops download ===\n")
	fmt.Printf("Repo:      %s\n", downloadRepo)
	fmt.Printf("Quant:     %s\n", downloadQuant)
	fmt.Printf("Pattern:   %s\n", pattern)
	fmt.Printf("Directory: %s\n\n", downloadDir)

	// Check if model already exists
	entries, err := os.ReadDir(downloadDir)
	if err == nil {
		for _, e := range entries {
			if strings.Contains(e.Name(), downloadQuant) && strings.HasSuffix(e.Name(), ".gguf") {
				info, _ := e.Info()
				if info != nil {
					sizeGB := float64(info.Size()) / (1024 * 1024 * 1024)
					fmt.Printf("Model already exists: %s (%.1f GB)\n", e.Name(), sizeGB)
					fmt.Printf("Delete it first to re-download.\n")
					return nil
				}
			}
		}
	}

	// Build the download command
	// huggingface-cli download unsloth/Qwen3.6-27B-GGUF --include '*Q4_K_M*' --local-dir ~/models
	dlArgs := []string{
		"download",
		downloadRepo,
		"--include", pattern,
		"--local-dir", downloadDir,
	}

	fmt.Printf("Running: %s %s\n\n", hfBin, strings.Join(dlArgs, " "))

	proc := exec.Command(hfBin, dlArgs...)
	proc.Stdout = os.Stdout
	proc.Stderr = os.Stderr
	proc.Stdin = os.Stdin

	if err := proc.Run(); err != nil {
		return fmt.Errorf("download failed: %w", err)
	}

	fmt.Printf("\nDownload complete. Model saved to %s\n", downloadDir)
	return nil
}
