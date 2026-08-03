# Dependency licence policy

CI inventories exact dependency versions from committed lockfiles with
OSV-Scanner, then evaluates every dependency according to how this repository is
deployed:

- `SERVE`: cloud-hosted software that is not distributed to customers.
- `SHIP`: containers, on-premises installs, and other distributed applications.
- `PUBLISH`: public source repositories, SDKs, and published packages.

The machine-readable rules are in `.license-policy.json`. Runtime and optional
dependencies are gated. Dev-only dependencies are classified and reported but
do not fail CI.

## Rollout and baseline

`.license-policy-baseline.json` records existing blocking findings by package,
version, detected licence expression, source lockfile, and deployment context.
CI fails only when a blocking finding has a fingerprint that is not in that
baseline. Updating a dependency version causes it to be evaluated again.

Unknown and unparseable licence metadata is currently warning-only during the
initial rollout. After the unknowns have been triaged, change
`rollout.unknown` from `warn` to `deny` to make the checker fully fail closed.
Explicit `LicenseRef-*` terms and the global deny list already fail closed.

## Exceptions and OR elections

Exceptions belong in `.license-policy-exceptions.json` and require all of these
fields:

```json
{
  "package": "package-name",
  "version": "1.2.3",
  "license": "LicenseRef-Example",
  "reason": "Why this exact version is acceptable",
  "approved_by": "Approver name or ticket",
  "review_by": "2027-01-31"
}
```

The checker rejects expired exceptions. For an `A OR B` expression, an active
entry whose `license` is exactly `A` or `B` records the elected option. Without
an election, the most restrictive option is treated as live.

## Obligations and vendored code

Apache-2.0 remains allowed, but shipped and published uses receive an
informational reminder to propagate applicable NOTICE content. Likely vendored
source directories and git submodules are reported separately and require a
tracked licence or a reviewed, expiring exception.

The LGPL linking boundary, GPL hosted-use boundary, weak-copyleft modification
analysis, attribution details, and licence compatibility still require counsel
review. Automated classification is factual compliance support, not legal
advice.
