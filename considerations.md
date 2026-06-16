# Considerations

A running log of stack/framework/tooling decisions for RepoRelay.
Each entry is a short, dated record: what we chose, why, and what
we gave up. New entries go at the top, just under the `---` divider.

---

## 2026-06-16 Discovery site: Astro vs Next.js

**Decision**: Astro
**Context**: The discovery site is one of four product surfaces. Mostly read-only — search input, results cards, topic pages, compare view. SEO is critical because the site is the public face of the product.
**Why**: Static-first + island model ships zero JS by default, which fits a content-and-SEO surface. The small interactivity we need (search, tabs, feedback) is exactly what islands are for. Per-page bundle is dramatically smaller than Next.js for this workload.
**Gave up**: Next.js's React ecosystem and easy path to a user dashboard. If we later need user accounts, saved recs, or an admin tool, that will be a separate Next.js app, not a rewrite of the discovery site.

---

## How to add a new entry

Paste this template just below the `---` divider (above the previous entry) and fill in:

```
## YYYY-MM-DD Short topic title

**Decision**:
**Context**:
**Why**:
**Gave up**:
```
