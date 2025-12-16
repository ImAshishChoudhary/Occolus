# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in Occolus, please report it responsibly.

### How to Report

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email your findings to security@occolus.dev
3. Include detailed steps to reproduce the vulnerability
4. Provide any relevant proof-of-concept code

### What to Expect

- Acknowledgment within 48 hours
- Status update within 7 days
- We aim to patch critical vulnerabilities within 14 days

### Scope

The following are in scope:
- Authentication and authorization flaws
- Injection vulnerabilities (SQL, command, etc.)
- Cross-site scripting (XSS)
- Cross-site request forgery (CSRF)
- Insecure data exposure
- Server-side request forgery (SSRF)

The following are out of scope:
- Denial of service attacks
- Social engineering
- Physical security
- Third-party dependencies (report to upstream)

### Safe Harbor

We will not pursue legal action against researchers who:
- Act in good faith
- Avoid privacy violations
- Do not destroy data
- Report findings promptly

## Security Best Practices

When deploying Occolus:

**API Keys**
- Store API keys in environment variables
- Never commit `.env` files to version control
- Rotate keys periodically

**Network**
- Use HTTPS in production
- Configure CORS appropriately
- Place behind a reverse proxy

**Dependencies**
- Keep dependencies updated
- Run `pip audit` and `npm audit` regularly
- Monitor for CVE announcements

## Contact

security@occolus.dev

