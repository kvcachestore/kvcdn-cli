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

// flyInstaller returns a container with the flyctl binary installed.
func flyInstaller() *dagger.Container {
	return dag.Container().
		From("alpine/curl").
		WithExec([]string{"sh", "-c", "curl -L https://fly.io/install.sh | sh"})
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

// Build compiles the kvcdn release binary, strips it, and returns the directory
// containing the packaged artifact.
func (m *Kvcdn) Build(ctx context.Context, src *dagger.Directory) (*dagger.Directory, error) {
	builder := rustBuilder(src, "release").
		WithExec([]string{"cargo", "build", "--release", "--locked"}).
		WithExec([]string{"cp", "/src/target/release/kvcdn", "/tmp/kvcdn"}).
		WithExec([]string{"strip", "/tmp/kvcdn"}).
		WithExec([]string{"sh", "-c", "mkdir -p /out && cp /tmp/kvcdn /out/kvcdn-x86_64-unknown-linux-gnu && tar -czf /out/kvcdn-x86_64-unknown-linux-gnu.tar.gz -C /out kvcdn-x86_64-unknown-linux-gnu"})

	return builder.Directory("/out"), nil
}

// Sbom generates a SPDX JSON SBOM for the release tarball.
func (m *Kvcdn) Sbom(ctx context.Context, artifact *dagger.File) (*dagger.File, error) {
	out, err := dag.Container().
		From(syftImage).
		WithFile("/input.tar.gz", artifact).
		WithExec([]string{"/syft", "/input.tar.gz", "-o", "spdx-json=/out/kvcdn-x86_64-unknown-linux-gnu.sbom.json"}).
		File("/out/kvcdn-x86_64-unknown-linux-gnu.sbom.json").
		Sync(ctx)
	return out, err
}

// Scan extracts the release tarball and runs Trivy against the contents,
// failing on CRITICAL/HIGH CVEs or secrets.
func (m *Kvcdn) Scan(ctx context.Context, artifact *dagger.File) error {
	_, err := dag.Container().
		From(trivyImage).
		WithFile("/input/kvcdn-x86_64-unknown-linux-gnu.tar.gz", artifact).
		WithWorkdir("/input").
		WithExec([]string{
			"sh", "-c",
			"tar -xzf kvcdn-x86_64-unknown-linux-gnu.tar.gz && rm kvcdn-x86_64-unknown-linux-gnu.tar.gz",
		}).
		WithExec([]string{
			"trivy", "fs", "--severity", "CRITICAL,HIGH",
			"--exit-code", "1", "--no-progress",
			".",
		}).
		Sync(ctx)
	return err
}

// Sign signs the release tarball and its SBOM with cosign using the provided
// private key secret. The key secret should be a cosign private key.
// It returns a directory containing the two .sig files.
func (m *Kvcdn) Sign(
	ctx context.Context,
	artifact *dagger.File,
	sbom *dagger.File,
	cosignKey *dagger.Secret,
) (*dagger.Directory, error) {
	const (
		tarballName = "kvcdn-x86_64-unknown-linux-gnu.tar.gz"
		sbomName    = "kvcdn-x86_64-unknown-linux-gnu.sbom.json"
	)

	signer := dag.Container().
		From(cosignImage).
		WithFile("/artifacts/"+tarballName, artifact).
		WithFile("/artifacts/"+sbomName, sbom).
		WithSecretVariable("COSIGN_PASSWORD", dag.SetSecret("cosign-password", "")).
		WithSecretVariable("COSIGN_PRIVATE_KEY", cosignKey).
		WithWorkdir("/artifacts").
		WithExec([]string{"cosign", "sign-blob", "--key", "env://COSIGN_PRIVATE_KEY", "--yes", tarballName}).
		WithExec([]string{"cosign", "sign-blob", "--key", "env://COSIGN_PRIVATE_KEY", "--yes", sbomName})

	if _, err := signer.Sync(ctx); err != nil {
		return nil, fmt.Errorf("cosign sign-blob: %w", err)
	}
	return signer.Directory("/artifacts"), nil
}

// Release runs PR checks, builds the release tarball, generates an SBOM,
// scans for vulnerabilities, and optionally signs artifacts if a cosign key is
// provided. It returns a directory containing the tarball, SBOM, signatures,
// and the extracted binary.
func (m *Kvcdn) Release(
	ctx context.Context,
	src *dagger.Directory,
	// Cosign private key secret for signing the tarball and SBOM.
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

	tarball := out.File("kvcdn-x86_64-unknown-linux-gnu.tar.gz")

	var sbomFile *dagger.File
	var scanErr error
	if err := errgroup(
		func() error {
			sbomFile, err = m.Sbom(ctx, tarball)
			return err
		},
		func() error {
			scanErr = m.Scan(ctx, tarball)
			return scanErr
		},
	); err != nil {
		return nil, fmt.Errorf("sbom/scan: %w", err)
	}

	out = out.WithFile("kvcdn-x86_64-unknown-linux-gnu.sbom.json", sbomFile)

	if cosignKey != nil {
		sigs, err := m.Sign(ctx, tarball, sbomFile, cosignKey)
		if err != nil {
			return nil, fmt.Errorf("sign: %w", err)
		}
		out = out.WithDirectory("/", sigs)
	}

	return out, nil
}

// DeployBackend builds the backend Docker image and deploys it to Fly.io.
// It expects a FLY_API_TOKEN secret to be available in the Dagger environment.
// Fly secrets (KVCDN_S3_*, KVCDN_CONTROL_BUCKET, etc.) are managed out-of-band
// with `fly secrets set` before calling this pipeline.
func (m *Kvcdn) DeployBackend(ctx context.Context, src *dagger.Directory, flyApiToken *dagger.Secret) (*dagger.Container, error) {
	backend := src.Directory("backend")

	deployer := dag.Container().
		From("node:22-alpine").
		WithDirectory("/backend", backend).
		WithWorkdir("/backend").
		WithFile("/usr/local/bin/flyctl", flyInstaller().File("/root/.fly/bin/flyctl")).
		WithSecretVariable("FLY_API_TOKEN", flyApiToken).
		WithExec([]string{"sh", "-c", "flyctl deploy --app kvcachestore --dockerfile Dockerfile"})

	return deployer.Sync(ctx)
}

// FlyCommand runs an arbitrary flyctl command in a container with the deploy token.
// Useful for checking status, reading logs, or inspecting secrets when the local
// CLI token is not available.
func (m *Kvcdn) FlyCommand(ctx context.Context, flyApiToken *dagger.Secret, args []string) (string, error) {
	cmd := append([]string{"flyctl"}, args...)
	out, err := dag.Container().
		From("alpine:latest").
		WithFile("/usr/local/bin/flyctl", flyInstaller().File("/root/.fly/bin/flyctl")).
		WithSecretVariable("FLY_API_TOKEN", flyApiToken).
		WithExec(cmd).
		Stdout(ctx)
	return out, err
}
