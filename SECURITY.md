# Security Policy

TSDynamics is a scientific Python library. While it is not a network service,
we take supply-chain integrity and the correctness of any code it executes
seriously, and we follow a **coordinated (responsible) disclosure** process for
security issues.

## Supported versions

Security fixes are applied to the latest released minor version on PyPI. We
recommend always running the most recent release.

| Version | Supported          |
| ------- | ------------------ |
| 3.1.x   | :white_check_mark: |
| < 3.1   | :x:                |

Because releases are
[fully automated](CONTRIBUTING.md#release-process) via
python-semantic-release, a fix merged to `main` ships in the next patch release
within minutes — so "upgrade to the latest release" is almost always the
remediation.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
pull requests, or discussions.**

Instead, report them privately through either of the following:

1. **GitHub private vulnerability reporting** (preferred): use the
   ["Report a vulnerability"](https://github.com/El3ssar/TSDynamics/security/advisories/new)
   button under the repository's **Security** tab. This opens a private
   advisory visible only to you and the maintainers.
2. **Email**: send a description to the maintainer at
   <kemossabee@gmail.com> with `SECURITY` in the subject line.

Please include, as far as you are able:

- A description of the issue and the impact you believe it has.
- The affected version(s) and platform (OS, architecture, Python version).
- A minimal reproducer (a short Python script, or the relevant input) plus the
  expected versus observed behaviour.
- Any known mitigations or workarounds.

## What to expect

We are a small, volunteer-maintained project, but we aim to:

- **Acknowledge** your report within **5 business days**.
- Provide an initial **assessment** (severity and whether we can reproduce it)
  within **10 business days**.
- Keep you informed of progress toward a fix and coordinate a disclosure
  timeline with you. We ask for a reasonable embargo period (typically up to
  **90 days**) so a fix can be released before details are made public.
- Credit you in the published advisory and release notes, unless you prefer to
  remain anonymous.

If a report is accepted, we will prepare a fix, publish a
[GitHub Security Advisory](https://github.com/El3ssar/TSDynamics/security/advisories),
and cut a release. If a report is declined, we will explain why.

## Scope

In scope:

- The published `tsdynamics` package (the pure-Python library and the compiled
  `tsdynamics._rust` extension).
- Code-execution or memory-safety issues reachable from documented public APIs
  (for example, malformed input that triggers undefined behaviour in the Rust
  engine rather than a clean Python exception).
- Supply-chain integrity of our release pipeline.

Out of scope:

- Vulnerabilities in third-party dependencies that are already publicly known
  and tracked upstream — though we still appreciate a heads-up, and our
  dependency audit (below) is designed to catch these automatically.
- Denial of service from intentionally pathological inputs (for example, asking
  the integrator for an astronomically large number of steps): this is expected
  resource usage, not a vulnerability.

## Supply-chain and dependency auditing

Security is enforced continuously in CI, not only on report:

- **Dependency vulnerability audit.** Our
  [`security-audit.yml`](.github/workflows/security-audit.yml) workflow runs
  `pip-audit` (Python dependencies) and `cargo-audit` (the Rust engine and its
  crates) and fails the build if a dependency with a known advisory is pulled
  in. It also runs on a schedule so a newly disclosed advisory in an existing
  dependency is surfaced even without a code change.
- **Automated dependency updates.** [Dependabot](.github/dependabot.yml) opens
  pull requests for outdated Python and GitHub-Actions dependencies on a monthly
  cadence.
- **Trusted publishing.** Releases are published to PyPI via OIDC
  [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no
  long-lived API tokens are stored in the repository.

Thank you for helping keep TSDynamics and its users safe.
