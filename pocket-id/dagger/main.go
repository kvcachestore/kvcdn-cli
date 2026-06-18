package main

import (
	"context"

	"dagger/pocket-id/internal/dagger"
)

type PocketId struct{}

const pocketIdApp = "kvcachestore-pocket-id"

// flyInstaller returns a container with the flyctl binary installed.
func flyInstaller() *dagger.Container {
	return dag.Container().
		From("alpine/curl").
		WithExec([]string{"sh", "-c", "curl -L https://fly.io/install.sh | sh"})
}

// pocketIdImageBuilder builds a custom Pocket ID image that seeds the
// smtpFrom config so admin login-code emails are sent from the verified
// Resend domain.
func pocketIdImageBuilder(src *dagger.Directory) *dagger.Container {
	return dag.Container().
		From("ghcr.io/pocket-id/pocket-id:latest").
		WithFile("/usr/local/bin/kvcdn-pocket-id-entrypoint.sh", src.File("entrypoint.sh")).
		WithExec([]string{"chmod", "+x", "/usr/local/bin/kvcdn-pocket-id-entrypoint.sh"}).
		WithExec([]string{"apk", "add", "--no-cache", "sqlite"}).
		WithEntrypoint([]string{"/usr/local/bin/kvcdn-pocket-id-entrypoint.sh"}).
		WithDefaultArgs([]string{"/app/pocket-id"})
}

// deployApp deploys a Fly.io app from a fly.toml directory.
func deployApp(ctx context.Context, name string, src *dagger.Directory, token *dagger.Secret) (*dagger.Container, error) {
	return dag.Container().
		From("node:22-alpine").
		WithDirectory("/app", src).
		WithWorkdir("/app").
		WithFile("/usr/local/bin/flyctl", flyInstaller().File("/root/.fly/bin/flyctl")).
		WithSecretVariable("FLY_API_TOKEN", token).
		WithExec([]string{"sh", "-c", "flyctl deploy --app " + name}).
		Sync(ctx)
}

// DeployPocketId builds the custom Pocket ID image and deploys it to Fly.io.
// It expects a FLY_API_TOKEN secret to be available in the Dagger environment.
func (m *PocketId) DeployPocketId(ctx context.Context, src *dagger.Directory, flyApiToken *dagger.Secret) (*dagger.Container, error) {
	return deployApp(ctx, pocketIdApp, src.Directory("pocket-id"), flyApiToken)
}

// DeployTunnel deploys the Cloudflare tunnel sidecar to Fly.io.
// It expects a FLY_API_TOKEN secret to be available in the Dagger environment.
func (m *PocketId) DeployTunnel(ctx context.Context, src *dagger.Directory, flyApiToken *dagger.Secret) (*dagger.Container, error) {
	return deployApp(ctx, "kvcachestore-pocket-id-cloudflared", src.Directory("pocket-id/cloudflared"), flyApiToken)
}

// DeployAll deploys the custom Pocket ID app and its Cloudflare tunnel sidecar.
// It expects a FLY_API_TOKEN secret to be available in the Dagger environment.
func (m *PocketId) DeployAll(ctx context.Context, src *dagger.Directory, flyApiToken *dagger.Secret) error {
	if _, err := m.DeployPocketId(ctx, src, flyApiToken); err != nil {
		return err
	}
	_, err := m.DeployTunnel(ctx, src, flyApiToken)
	return err
}
