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
	patchesVllmDir string
)

var patchesCmd = &cobra.Command{
	Use:   "patches",
	Short: "List and apply vLLM/llama.cpp patches",
}

var patchesListCmd = &cobra.Command{
	Use:   "list",
	Short: "List available vLLM/llama.cpp patches",
	RunE:  runPatchesList,
}

var patchesApplyCmd = &cobra.Command{
	Use:   "apply",
	Short: "Apply vLLM patches to a vLLM checkout",
	RunE:  runPatchesApply,
}

func init() {
	home, _ := os.UserHomeDir()

	patchesApplyCmd.Flags().StringVar(&patchesVllmDir, "vllm-dir", filepath.Join(home, "vllm"), "Path to vLLM checkout")

	patchesCmd.AddCommand(patchesListCmd)
	patchesCmd.AddCommand(patchesApplyCmd)
}

func listPatchDir(label, dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Printf("  %s: (directory not found)\n", label)
			return
		}
		fmt.Printf("  %s: error reading: %v\n", label, err)
		return
	}

	var patches []string
	for _, e := range entries {
		name := e.Name()
		if strings.HasSuffix(name, ".patch") || strings.HasSuffix(name, ".diff") {
			patches = append(patches, name)
		}
	}

	if len(patches) == 0 {
		fmt.Printf("  %s: (no patches)\n", label)
		fmt.Printf("    Directory: %s\n", dir)
	} else {
		fmt.Printf("  %s: %d patch(es)\n", label, len(patches))
		for _, p := range patches {
			fmt.Printf("    - %s\n", p)
		}
	}
}

func runPatchesList(cmd *cobra.Command, args []string) error {
	home, _ := os.UserHomeDir()

	fmt.Println("=== Available Patches ===")
	fmt.Println()

	fmt.Println("llama.cpp patches:")
	listPatchDir("Infrastructure", filepath.Join(home, "qwen-ops", "llamacpp", "infrastructure"))
	listPatchDir("Optimizations", filepath.Join(home, "qwen-ops", "llamacpp", "optimizations"))
	fmt.Println()

	fmt.Println("vLLM patches:")
	listPatchDir("Patches", filepath.Join(home, "qwen-ops", "vllm", "patches"))
	listPatchDir("Optimizations", filepath.Join(home, "qwen-ops", "vllm", "optimizations"))
	fmt.Println()

	fmt.Println("To apply llama.cpp patches: qwen-ops mtp apply [--llama-cpp-dir PATH]")
	fmt.Println("To apply vLLM patches:      qwen-ops patches apply [--vllm-dir PATH]")

	return nil
}

func runPatchesApply(cmd *cobra.Command, args []string) error {
	// Verify vLLM checkout exists
	if _, err := os.Stat(filepath.Join(patchesVllmDir, "setup.py")); err != nil {
		// Also check for pyproject.toml
		if _, err := os.Stat(filepath.Join(patchesVllmDir, "pyproject.toml")); err != nil {
			return fmt.Errorf("vLLM not found at %s (missing setup.py/pyproject.toml)", patchesVllmDir)
		}
	}

	home, _ := os.UserHomeDir()
	patchDir := filepath.Join(home, "qwen-ops", "vllm", "patches")
	optimDir := filepath.Join(home, "qwen-ops", "vllm", "optimizations")

	patches, err := collectPatches(patchDir)
	if err != nil {
		return err
	}

	optimPatches, err := collectPatches(optimDir)
	if err != nil {
		return err
	}

	allPatches := append(patches, optimPatches...)

	fmt.Printf("=== vLLM Patch Application ===\n")
	fmt.Printf("Target: %s\n\n", patchesVllmDir)

	if len(allPatches) == 0 {
		fmt.Printf("No patches found.\n")
		fmt.Printf("Add .patch or .diff files to:\n")
		fmt.Printf("  %s\n", patchDir)
		fmt.Printf("  %s\n", optimDir)
		return nil
	}

	for _, p := range allPatches {
		fmt.Printf("  Applying: %s\n", filepath.Base(p))

		gitCmd := exec.Command("git", "apply", "--check", p)
		gitCmd.Dir = patchesVllmDir
		if err := gitCmd.Run(); err != nil {
			fmt.Printf("    WARNING: patch may not apply cleanly, trying with --3way\n")
			gitCmd = exec.Command("git", "apply", "--3way", p)
		} else {
			gitCmd = exec.Command("git", "apply", p)
		}
		gitCmd.Dir = patchesVllmDir
		gitCmd.Stdout = os.Stdout
		gitCmd.Stderr = os.Stderr
		if err := gitCmd.Run(); err != nil {
			return fmt.Errorf("failed to apply %s: %w", filepath.Base(p), err)
		}
		fmt.Printf("    OK\n")
	}

	fmt.Printf("\nDone. %d patch(es) applied to %s\n", len(allPatches), patchesVllmDir)
	return nil
}
