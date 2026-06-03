# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Do not open a public issue for security vulnerabilities.

### Preferred: private reporting

If [Private vulnerability reporting](https://github.com/alirazaanis/vitalroute/security/policy#reporting-a-vulnerability)
is enabled on the repository, use **Report a vulnerability** on the Security tab.
Reports are visible only to maintainers until disclosed.

### Alternative: security advisory

Submit a [private security advisory](https://github.com/alirazaanis/vitalroute/security/advisories/new)
via GitHub Security Advisories.

### Include in your report

- Description of the vulnerability
- Steps to reproduce
- Impact assessment (if known)
- Suggested fix (optional)

Maintainers aim to acknowledge reports within 7 days.

## Automated security checks

This repository uses:

| Check | Configuration |
|---|---|
| Secret scanning | Enabled by GitHub (push-time detection) |
| Dependabot | `.github/dependabot.yml` (weekly pip and Actions updates) |
| CodeQL | `.github/workflows/codeql.yml` (Python analysis on push/PR) |
| CI tests | `.github/workflows/ci.yml` |
