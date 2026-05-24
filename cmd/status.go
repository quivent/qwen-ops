package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show what's installed/running on current machine",
	Long: `Check the current state of the Qwen MTP inference stack:
  - Is llama.cpp built? What version?
  - Is a model downloaded? Which?
  - Is llama-server running? PID, port?
  - GPU info (nvidia-smi)`,
	RunE: runStatus,
}

func runStatus(cmd *cobra.Command, args []string) error {
	home, _ := os.UserHomeDir()

	fmt.Println("=== qwen-ops status ===")
	fmt.Println()

	// 1. llama.cpp build status
	checkLlamaCpp(home)
	fmt.Println()

	// 2. Model status
	checkModels(home)
	fmt.Println()

	// 3. Server status
	checkServer()
	fmt.Println()

	// 4. GPU info
	checkGPU()
	fmt.Println()

	// 5. Patch inventory
	checkPatches(home)

	return nil
}

func checkLlamaCpp(home string) {
	fmt.Println("[llama.cpp]")
	llamaDir := filepath.Join(home, "llama.cpp")

	if _, err := os.Stat(llamaDir); err != nil {
		fmt.Println("  Not found at ~/llama.cpp")
		return
	}

	fmt.Printf("  Directory: %s\n", llamaDir)

	// Check if built
	serverBin := filepath.Join(llamaDir, "build", "bin", "llama-server")
	if _, err := os.Stat(serverBin); err == nil {
		fmt.Printf("  llama-server: %s\n", serverBin)
	} else {
		fmt.Println("  llama-server: not built")
	}

	cliBin := filepath.Join(llamaDir, "build", "bin", "llama-cli")
	if _, err := os.Stat(cliBin); err == nil {
		fmt.Printf("  llama-cli: %s\n", cliBin)
	}

	benchBin := filepath.Join(llamaDir, "build", "bin", "llama-bench")
	if _, err := os.Stat(benchBin); err == nil {
		fmt.Printf("  llama-bench: %s\n", benchBin)
	}

	// Git version
	gitCmd := exec.Command("git", "log", "--oneline", "-1")
	gitCmd.Dir = llamaDir
	out, err := gitCmd.Output()
	if err == nil {
		fmt.Printf("  Version: %s", string(out))
	}

	// Check for MTP-related modifications
	gitCmd = exec.Command("git", "diff", "--stat")
	gitCmd.Dir = llamaDir
	out, err = gitCmd.Output()
	if err == nil && len(out) > 0 {
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		fmt.Printf("  Modified files: %d\n", len(lines))
	}
}

func checkModels(home string) {
	fmt.Println("[Models]")
	modelsDir := filepath.Join(home, "models")

	if _, err := os.Stat(modelsDir); err != nil {
		fmt.Println("  No ~/models directory")
		return
	}

	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		fmt.Printf("  Error reading models dir: %v\n", err)
		return
	}

	found := false
	for _, e := range entries {
		name := e.Name()
		if strings.HasSuffix(name, ".gguf") {
			info, err := e.Info()
			if err == nil {
				sizeGB := float64(info.Size()) / (1024 * 1024 * 1024)
				fmt.Printf("  %s (%.1f GB)\n", name, sizeGB)
			} else {
				fmt.Printf("  %s\n", name)
			}
			found = true
		}
	}

	if !found {
		fmt.Println("  No .gguf models found in ~/models")
		fmt.Println("  Run: qwen-ops download")
	}
}

func checkServer() {
	fmt.Println("[Server]")

	// Check for running llama-server processes
	out, err := exec.Command("pgrep", "-la", "llama-server").Output()
	if err != nil || len(out) == 0 {
		fmt.Println("  llama-server: not running")
		return
	}

	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	for _, line := range lines {
		parts := strings.SplitN(line, " ", 2)
		if len(parts) >= 1 {
			pid := parts[0]
			fmt.Printf("  llama-server: running (PID %s)\n", pid)
		}
	}

	// Try to detect port from process args
	out, err = exec.Command("ps", "aux").Output()
	if err == nil {
		for _, line := range strings.Split(string(out), "\n") {
			if strings.Contains(line, "llama-server") && !strings.Contains(line, "grep") {
				if idx := strings.Index(line, "--port"); idx >= 0 {
					rest := line[idx+7:]
					port := strings.Fields(rest)
					if len(port) > 0 {
						fmt.Printf("  Port: %s\n", port[0])
					}
				}
			}
		}
	}
}

func checkGPU() {
	fmt.Println("[GPU]")

	// Try nvidia-smi
	out, err := exec.Command("nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader").Output()
	if err == nil && len(out) > 0 {
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		for i, line := range lines {
			fmt.Printf("  GPU %d: %s\n", i, strings.TrimSpace(line))
		}
		return
	}

	// Try nvidia-smi without query format (some versions don't support it)
	out, err = exec.Command("nvidia-smi").Output()
	if err == nil {
		// Just show the first few relevant lines
		lines := strings.Split(string(out), "\n")
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if strings.Contains(trimmed, "NVIDIA") || strings.Contains(trimmed, "Driver") || strings.Contains(trimmed, "MiB") {
				fmt.Printf("  %s\n", trimmed)
			}
		}
		return
	}

	// Check for Apple Silicon (macOS)
	out, err = exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
	if err == nil {
		fmt.Printf("  CPU: %s", string(out))
	}

	out, err = exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err == nil {
		memStr := strings.TrimSpace(string(out))
		// Try to convert to GB
		var memBytes int64
		fmt.Sscanf(memStr, "%d", &memBytes)
		if memBytes > 0 {
			fmt.Printf("  Memory: %.0f GB (unified)\n", float64(memBytes)/(1024*1024*1024))
		}
	}

	fmt.Println("  Note: No NVIDIA GPU detected; running on CPU/Apple Silicon")
}

func checkPatches(home string) {
	fmt.Println("[Patches]")

	dirs := []struct {
		label string
		path  string
	}{
		{"llama.cpp/infrastructure", filepath.Join(home, "qwen-ops", "llamacpp", "infrastructure")},
		{"llama.cpp/optimizations", filepath.Join(home, "qwen-ops", "llamacpp", "optimizations")},
		{"vllm/patches", filepath.Join(home, "qwen-ops", "vllm", "patches")},
		{"vllm/optimizations", filepath.Join(home, "qwen-ops", "vllm", "optimizations")},
	}

	for _, d := range dirs {
		entries, err := os.ReadDir(d.path)
		if err != nil {
			fmt.Printf("  %s: (not found)\n", d.label)
			continue
		}
		count := 0
		for _, e := range entries {
			if strings.HasSuffix(e.Name(), ".patch") || strings.HasSuffix(e.Name(), ".diff") {
				count++
			}
		}
		fmt.Printf("  %s: %d patch(es)\n", d.label, count)
	}
}
