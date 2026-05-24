package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/spf13/cobra"
)

var (
	serveModel    string
	servePort     int
	serveHost     string
	serveCtx      int
	serveBatch    int
	serveUBatch   int
	serveThreads  int
	serveParallel int
	serveSpec     bool
	serveSpecN    int
	serveLlamaBin string
	serveDryRun   bool
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Launch optimized Qwen3.6 inference via llama-server",
	Long: `Launch llama-server with proven GH200 optimal flags for Qwen3.6-27B.

Default flags: -ngl 99 -fa on -c 8192 -b 2048 -ub 2048 -t 34
  --mlock --no-mmap -ctk q8_0 -ctv q4_0 --parallel 8
  --cont-batching --host 0.0.0.0 --port 8080

With --spec: adds --spec-type draft-mtp --spec-draft-n-max 3`,
	RunE: runServe,
}

func init() {
	home, _ := os.UserHomeDir()

	serveCmd.Flags().StringVarP(&serveModel, "model", "m", filepath.Join(home, "models", "Qwen3.6-27B-Q4_K_M.gguf"), "Path to GGUF model file")
	serveCmd.Flags().IntVar(&servePort, "port", 8080, "Server port")
	serveCmd.Flags().StringVar(&serveHost, "host", "0.0.0.0", "Server host")
	serveCmd.Flags().IntVarP(&serveCtx, "ctx", "c", 8192, "Context length")
	serveCmd.Flags().IntVarP(&serveBatch, "batch", "b", 2048, "Batch size")
	serveCmd.Flags().IntVar(&serveUBatch, "ubatch", 2048, "Micro-batch size")
	serveCmd.Flags().IntVarP(&serveThreads, "threads", "t", 34, "Number of threads")
	serveCmd.Flags().IntVar(&serveParallel, "parallel", 8, "Number of parallel slots")
	serveCmd.Flags().BoolVar(&serveSpec, "spec", false, "Enable speculative decoding (MTP)")
	serveCmd.Flags().IntVar(&serveSpecN, "spec-n", 3, "Max speculative draft tokens")
	serveCmd.Flags().StringVar(&serveLlamaBin, "llama-server", "", "Path to llama-server binary (auto-detected if empty)")
	serveCmd.Flags().BoolVar(&serveDryRun, "dry-run", false, "Print the command without executing")
}

func findLlamaServer() (string, error) {
	if serveLlamaBin != "" {
		if _, err := os.Stat(serveLlamaBin); err != nil {
			return "", fmt.Errorf("llama-server not found at %s", serveLlamaBin)
		}
		return serveLlamaBin, nil
	}

	// Check common locations
	home, _ := os.UserHomeDir()
	candidates := []string{
		filepath.Join(home, "llama.cpp", "build", "bin", "llama-server"),
		filepath.Join(home, "llama.cpp", "llama-server"),
		"/usr/local/bin/llama-server",
	}

	// Also check PATH
	if p, err := exec.LookPath("llama-server"); err == nil {
		return p, nil
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}

	return "", fmt.Errorf("llama-server not found; use --llama-server to specify path")
}

func runServe(cmd *cobra.Command, args []string) error {
	serverBin, err := findLlamaServer()
	if err != nil {
		return err
	}

	// Check model exists
	if _, err := os.Stat(serveModel); err != nil {
		return fmt.Errorf("model not found at %s\nRun: qwen-ops download", serveModel)
	}

	// Build argument list with proven GH200 optimal flags
	serverArgs := []string{
		"-m", serveModel,
		"-ngl", "99",
		"-fa", "on",
		"-c", fmt.Sprintf("%d", serveCtx),
		"-b", fmt.Sprintf("%d", serveBatch),
		"-ub", fmt.Sprintf("%d", serveUBatch),
		"-t", fmt.Sprintf("%d", serveThreads),
		"--mlock",
		"--no-mmap",
		"-ctk", "q8_0",
		"-ctv", "q4_0",
		"--parallel", fmt.Sprintf("%d", serveParallel),
		"--cont-batching",
		"--host", serveHost,
		"--port", fmt.Sprintf("%d", servePort),
	}

	// Add speculative decoding flags if requested
	if serveSpec {
		serverArgs = append(serverArgs,
			"--spec-type", "draft-mtp",
			"--spec-draft-n-max", fmt.Sprintf("%d", serveSpecN),
		)
	}

	if serveDryRun {
		fmt.Printf("%s", serverBin)
		for _, a := range serverArgs {
			fmt.Printf(" %s", a)
		}
		fmt.Println()
		return nil
	}

	mode := "standard"
	if serveSpec {
		mode = fmt.Sprintf("MTP speculative (n=%d)", serveSpecN)
	}
	fmt.Printf("=== qwen-ops serve ===\n")
	fmt.Printf("Mode:   %s\n", mode)
	fmt.Printf("Model:  %s\n", serveModel)
	fmt.Printf("Server: %s\n", serverBin)
	fmt.Printf("Listen: %s:%d\n", serveHost, servePort)
	fmt.Printf("Ctx:    %d | Batch: %d | Threads: %d | Parallel: %d\n", serveCtx, serveBatch, serveThreads, serveParallel)
	fmt.Println()

	proc := exec.Command(serverBin, serverArgs...)
	proc.Stdout = os.Stdout
	proc.Stderr = os.Stderr
	proc.Stdin = os.Stdin

	return proc.Run()
}
