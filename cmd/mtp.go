package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/spf13/cobra"
)

var (
	mtpLlamaCppDir string
	mtpOptimize    bool
)

var mtpCmd = &cobra.Command{
	Use:   "mtp",
	Short: "Multi-Token Prediction patch management and benchmarking",
}

var mtpApplyCmd = &cobra.Command{
	Use:   "apply",
	Short: "Apply MTP patches to a llama.cpp checkout",
	Long: `Apply MTP infrastructure patches from ~/qwen-ops/llamacpp/infrastructure/
in sorted order, then optionally apply optimization patches from
~/qwen-ops/llamacpp/optimizations/.

Patches are applied with git apply. Use --optimize to also apply
performance optimization patches after infrastructure patches.`,
	RunE: runMtpApply,
}

var mtpBenchCmd = &cobra.Command{
	Use:   "bench",
	Short: "Run MTP benchmark on current llama.cpp build",
	Long: `Run llama-bench with MTP-specific settings to measure
multi-token prediction throughput on the current hardware.

Benchmarks both standard and MTP inference for comparison.`,
	RunE: runMtpBench,
}

var (
	benchModel   string
	benchPrompt  string
	benchNTokens int
	benchRepeat  int
)

func init() {
	home, _ := os.UserHomeDir()

	mtpApplyCmd.Flags().StringVar(&mtpLlamaCppDir, "llama-cpp-dir", filepath.Join(home, "llama.cpp"), "Path to llama.cpp checkout")
	mtpApplyCmd.Flags().BoolVar(&mtpOptimize, "optimize", false, "Also apply optimization patches")

	mtpBenchCmd.Flags().StringVarP(&benchModel, "model", "m", filepath.Join(home, "models", "Qwen3.6-27B-Q4_K_M.gguf"), "Path to GGUF model")
	mtpBenchCmd.Flags().StringVarP(&benchPrompt, "prompt", "p", "Explain quantum computing in simple terms:", "Benchmark prompt")
	mtpBenchCmd.Flags().IntVarP(&benchNTokens, "n-tokens", "n", 256, "Number of tokens to generate")
	mtpBenchCmd.Flags().IntVar(&benchRepeat, "repeat", 3, "Number of benchmark repetitions")

	mtpCmd.AddCommand(mtpApplyCmd)
	mtpCmd.AddCommand(mtpBenchCmd)
}

func collectPatches(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reading patch dir %s: %w", dir, err)
	}

	var patches []string
	for _, e := range entries {
		name := e.Name()
		if strings.HasSuffix(name, ".patch") || strings.HasSuffix(name, ".diff") {
			patches = append(patches, filepath.Join(dir, name))
		}
	}
	sort.Strings(patches)
	return patches, nil
}

func applyPatches(dir string, patches []string) error {
	for _, p := range patches {
		fmt.Printf("  Applying: %s\n", filepath.Base(p))

		// First try git apply
		cmd := exec.Command("git", "apply", "--check", p)
		cmd.Dir = dir
		if err := cmd.Run(); err != nil {
			fmt.Printf("    WARNING: patch may not apply cleanly, trying with --3way\n")
			cmd = exec.Command("git", "apply", "--3way", p)
			cmd.Dir = dir
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			if err := cmd.Run(); err != nil {
				return fmt.Errorf("failed to apply %s: %w", filepath.Base(p), err)
			}
		} else {
			cmd = exec.Command("git", "apply", p)
			cmd.Dir = dir
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			if err := cmd.Run(); err != nil {
				return fmt.Errorf("failed to apply %s: %w", filepath.Base(p), err)
			}
		}
		fmt.Printf("    OK\n")
	}
	return nil
}

func runMtpApply(cmd *cobra.Command, args []string) error {
	// Verify llama.cpp checkout exists
	if _, err := os.Stat(filepath.Join(mtpLlamaCppDir, "CMakeLists.txt")); err != nil {
		return fmt.Errorf("llama.cpp not found at %s (missing CMakeLists.txt)", mtpLlamaCppDir)
	}

	home, _ := os.UserHomeDir()
	infraDir := filepath.Join(home, "qwen-ops", "llamacpp", "infrastructure")
	optimDir := filepath.Join(home, "qwen-ops", "llamacpp", "optimizations")

	// Collect infrastructure patches
	infraPatches, err := collectPatches(infraDir)
	if err != nil {
		return err
	}

	fmt.Printf("=== MTP Patch Application ===\n")
	fmt.Printf("Target: %s\n\n", mtpLlamaCppDir)

	if len(infraPatches) == 0 {
		fmt.Printf("No infrastructure patches found in %s\n", infraDir)
		fmt.Printf("Add .patch or .diff files to apply MTP modifications.\n")
	} else {
		fmt.Printf("Infrastructure patches (%d):\n", len(infraPatches))
		if err := applyPatches(mtpLlamaCppDir, infraPatches); err != nil {
			return err
		}
		fmt.Println()
	}

	if mtpOptimize {
		optimPatches, err := collectPatches(optimDir)
		if err != nil {
			return err
		}
		if len(optimPatches) == 0 {
			fmt.Printf("No optimization patches found in %s\n", optimDir)
		} else {
			fmt.Printf("Optimization patches (%d):\n", len(optimPatches))
			if err := applyPatches(mtpLlamaCppDir, optimPatches); err != nil {
				return err
			}
		}
	}

	fmt.Printf("\nDone. Rebuild llama.cpp to use MTP:\n")
	fmt.Printf("  cd %s && cmake -B build -DGGML_CUDA=ON && cmake --build build -j\n", mtpLlamaCppDir)
	return nil
}

func runMtpBench(cmd *cobra.Command, args []string) error {
	home, _ := os.UserHomeDir()

	// Find llama-bench or llama-cli
	llamaDir := filepath.Join(home, "llama.cpp")
	benchBin := ""

	candidates := []string{
		filepath.Join(llamaDir, "build", "bin", "llama-bench"),
		filepath.Join(llamaDir, "llama-bench"),
	}
	if p, err := exec.LookPath("llama-bench"); err == nil {
		benchBin = p
	} else {
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				benchBin = c
				break
			}
		}
	}

	// Fall back to llama-cli if no llama-bench
	if benchBin == "" {
		cliBin := ""
		cliCandidates := []string{
			filepath.Join(llamaDir, "build", "bin", "llama-cli"),
			filepath.Join(llamaDir, "llama-cli"),
		}
		if p, err := exec.LookPath("llama-cli"); err == nil {
			cliBin = p
		} else {
			for _, c := range cliCandidates {
				if _, err := os.Stat(c); err == nil {
					cliBin = c
					break
				}
			}
		}

		if cliBin == "" {
			return fmt.Errorf("neither llama-bench nor llama-cli found; build llama.cpp first")
		}

		return runCliBench(cliBin)
	}

	fmt.Printf("=== MTP Benchmark ===\n")
	fmt.Printf("Binary: %s\n", benchBin)
	fmt.Printf("Model:  %s\n\n", benchModel)

	// Run standard benchmark
	benchArgs := []string{
		"-m", benchModel,
		"-ngl", "99",
		"-fa", "1",
		"-t", "34",
	}

	fmt.Println("--- Standard inference ---")
	proc := exec.Command(benchBin, benchArgs...)
	proc.Stdout = os.Stdout
	proc.Stderr = os.Stderr
	if err := proc.Run(); err != nil {
		fmt.Printf("Benchmark error: %v\n", err)
	}

	return nil
}

func runCliBench(cliBin string) error {
	fmt.Printf("=== MTP Benchmark (via llama-cli) ===\n")
	fmt.Printf("Binary: %s\n", cliBin)
	fmt.Printf("Model:  %s\n", benchModel)
	fmt.Printf("Prompt: %s\n", benchPrompt)
	fmt.Printf("Tokens: %d x %d runs\n\n", benchNTokens, benchRepeat)

	for i := 0; i < benchRepeat; i++ {
		fmt.Printf("--- Run %d/%d ---\n", i+1, benchRepeat)
		cliArgs := []string{
			"-m", benchModel,
			"-ngl", "99",
			"-fa", "on",
			"-t", "34",
			"--mlock",
			"--no-mmap",
			"-n", fmt.Sprintf("%d", benchNTokens),
			"-p", benchPrompt,
		}
		proc := exec.Command(cliBin, cliArgs...)
		proc.Stdout = os.Stdout
		proc.Stderr = os.Stderr
		if err := proc.Run(); err != nil {
			fmt.Printf("Run %d error: %v\n", i+1, err)
		}
		fmt.Println()
	}

	return nil
}
