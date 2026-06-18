## What

Briefly describe what changed and why.

## How

- High-level implementation approach.
- Any breaking changes or new dependencies.

## Testing

- How was this tested?
- Include command output if relevant (e.g., `cargo test --workspace`, `npm test`).

## Checklist

- [ ] `cargo test --workspace` passes.
- [ ] `cargo fmt --check` passes.
- [ ] `cargo clippy --all-targets -- -D warnings` passes.
- [ ] Backend tests pass (`cd backend && npm test`).
- [ ] Dagger PR checks pass (`dagger call -m dagger pr --src=.`).
- [ ] Dagger release build still works (`./scripts/build-release.sh`).
- [ ] README and other docs updated if user-facing behavior changed.
