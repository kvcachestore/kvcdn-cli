package main

import (
	"context"
	"fmt"
	"sync"

	"dagger/kvcdn/internal/dagger"
)

const (
	rustImage = "rust:1.96-bookworm"

	syftImage   = "anchore/syft:v1.45.1"
	trivyImage  = "aquasec/trivy:0.62.0"
	cosignImage = "cgr.dev/chainguard/cosign:v2.4.0"

	cargoRegistryCache = "kvcdn-cargo-registry"
	cargoGitCache      = "kvcdn-cargo-git"
	cargoDebugCache    = "kvcdn-cargo-target-debug"
	cargoReleaseCache  = "kvcdn-cargo-target-release"
	npmCache           = "kvcdn-npm-cache"
)

type Kvcdn struct{}

// withoutGit strips the .git directory from the source tree so that cache keys
// depend on the code rather than on Git metadata.
func withoutGit(src *dagger.Directory) *dagger.Directory {
	return src.WithoutDirectory(".git")
}

// rustBuilder returns a Rust container with the source and cargo caches mounted.
// targetTag selects the debug or release target cache.
func rustBuilder(src *dagger.Directory, targetTag string) *dagger.Container {
	cargoRegistry := dag.CacheVolume(cargoRegistryCache)
	cargoGit := dag.CacheVolume(cargoGitCache)
	targetCache := dag.CacheVolume("kvcdn-cargo-target-" + targetTag)

	return dag.Container().
		From(rustImage).
		WithMountedDirectory("/src", withoutGit(src)).
		WithWorkdir("/src").
		WithMountedCache("/usr/local/cargo/registry", cargoRegistry).
		WithMountedCache("/usr/local/cargo/git", cargoGit).
		WithMountedCache("/src/target", targetCache)
}

// Lint runs cargo fmt --check and cargo clippy.
func (m *Kvcdn) Lint(ctx context.Context, src *dagger.Directory) error {
	_, err := rustBuilder(src, "debug").
		WithExec([]string{"rustup", "component", "add", "rustfmt", "clippy"}).
		WithExec([]string{"cargo", "fmt", "--check"}).
		WithExec([]string{"cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"}).
		Sync(ctx)
	return err
}

// Test runs the Rust test suite.
func (m *Kvcdn) Test(ctx context.Context, src *dagger.Directory) error {
	_, err := rustBuilder(src, "debug").
		WithExec([]string{"cargo", "test", "--workspace"}).
		Sync(ctx)
	return err
}

// BackendPr runs the backend TypeScript test suite in a Node container.
func (m *Kvcdn) BackendPr(ctx context.Context, src *dagger.Directory) error {
	backend := src.Directory("backend")
	npm := dag.CacheVolume(npmCache)

	_, err := dag.Container().
		From("node:22").
		WithMountedDirectory("/backend", backend).
		WithWorkdir("/backend").
		WithMountedCache("/root/.npm", npm).
		WithExec([]string{"npm", "ci"}).
		WithExec([]string{"npm", "test"}).
		Sync(ctx)
	return err
}

// task is a single unit of work for the minimal errgroup.
type task func() error

// errgroup runs tasks concurrently and returns the first error encountered.
func errgroup(tasks ...task) error {
	var wg sync.WaitGroup
	errCh := make(chan error, len(tasks))

	for _, t := range tasks {
		wg.Add(1)
		go func(f task) {
			defer wg.Done()
			if err := f(); err != nil {
				errCh <- err
			}
		}(t)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		return err
	}
	return nil
}

// Pr runs lint, test, and backend PR checks in parallel.
func (m *Kvcdn) Pr(ctx context.Context, src *dagger.Directory) error {
	return errgroup(
		func() error { return m.Lint(ctx, src) },
		func() error { return m.Test(ctx, src) },
		func() error { return m.BackendPr(ctx, src) },
	)
}

const binaryName = "kvcdn-x86_64-unknown-linux-gnu"

// Build compiles the kvcdn release binary, strips it, and returns the directory
// containing the raw binary.
func (m *Kvcdn) Build(ctx context.Context, src *dagger.Directory) (*dagger.Directory, error) {
	builder := rustBuilder(src, "release").
		WithExec([]string{"cargo", "build", "--release", "--locked"}).
		WithExec([]string{"cp", "/src/target/release/kvcdn", "/tmp/" + binaryName}).
		WithExec([]string{"strip", "/tmp/" + binaryName}).
		WithExec([]string{"sh", "-c", "mkdir -p /out && cp /tmp/" + binaryName + " /out/" + binaryName})

	return builder.Directory("/out"), nil
}

// Sbom generates a SPDX JSON SBOM for the release binary.
func (m *Kvcdn) Sbom(ctx context.Context, artifact *dagger.File) (*dagger.File, error) {
	out, err := dag.Container().
		From(syftImage).
		WithFile("/input/"+binaryName, artifact).
		WithWorkdir("/input").
		WithExec([]string{"/syft", binaryName, "-o", "spdx-json=/out/" + binaryName + ".sbom.json"}).
		File("/out/" + binaryName + ".sbom.json").
		Sync(ctx)
	return out, err
}

// Scan runs Trivy against the release binary, failing on CRITICAL/HIGH CVEs or secrets.
func (m *Kvcdn) Scan(ctx context.Context, artifact *dagger.File) error {
	_, err := dag.Container().
		From(trivyImage).
		WithFile("/input/"+binaryName, artifact).
		WithWorkdir("/input").
		WithExec([]string{
			"trivy", "fs", "--severity", "CRITICAL,HIGH",
			"--exit-code", "1", "--no-progress",
			".",
		}).
		Sync(ctx)
	return err
}

// Sign signs the release binary and its SBOM with cosign using the provided
// private key secret. The key secret should be a cosign private key.
// It returns a directory containing the two .sig files.
func (m *Kvcdn) Sign(
	ctx context.Context,
	artifact *dagger.File,
	sbom *dagger.File,
	cosignKey *dagger.Secret,
) (*dagger.Directory, error) {
	const sbomName = binaryName + ".sbom.json"

	signer := dag.Container().
		From(cosignImage).
		WithFile("/artifacts/"+binaryName, artifact).
		WithFile("/artifacts/"+sbomName, sbom).
		WithSecretVariable("COSIGN_PASSWORD", dag.SetSecret("cosign-password", "")).
		WithSecretVariable("COSIGN_PRIVATE_KEY", cosignKey).
		WithWorkdir("/artifacts").
		WithExec([]string{"cosign", "sign-blob", "--key", "env://COSIGN_PRIVATE_KEY", "--yes", binaryName}).
		WithExec([]string{"cosign", "sign-blob", "--key", "env://COSIGN_PRIVATE_KEY", "--yes", sbomName})

	if _, err := signer.Sync(ctx); err != nil {
		return nil, fmt.Errorf("cosign sign-blob: %w", err)
	}
	return signer.Directory("/artifacts"), nil
}

// Release runs PR checks, builds the release binary, generates an SBOM,
// scans for vulnerabilities, and optionally signs artifacts if a cosign key is
// provided. It returns a directory containing the binary, SBOM, signatures,
// and the extracted binary.
func (m *Kvcdn) Release(
	ctx context.Context,
	src *dagger.Directory,
	// Cosign private key secret for signing the binary and SBOM.
	// +optional
	cosignKey *dagger.Secret,
) (*dagger.Directory, error) {
	if err := m.Pr(ctx, src); err != nil {
		return nil, fmt.Errorf("pr checks: %w", err)
	}

	out, err := m.Build(ctx, src)
	if err != nil {
		return nil, fmt.Errorf("build: %w", err)
	}

	binary := out.File(binaryName)

	var sbomFile *dagger.File
	var scanErr error
	if err := errgroup(
		func() error {
			sbomFile, err = m.Sbom(ctx, binary)
			return err
		},
		func() error {
			scanErr = m.Scan(ctx, binary)
			return scanErr
		},
	); err != nil {
		return nil, fmt.Errorf("sbom/scan: %w", err)
	}

	out = out.WithFile(binaryName+".sbom.json", sbomFile)

	if cosignKey != nil {
		sigs, err := m.Sign(ctx, binary, sbomFile, cosignKey)
		if err != nil {
			return nil, fmt.Errorf("sign: %w", err)
		}
		out = out.WithDirectory("/", sigs)
	}

	return out, nil
}

