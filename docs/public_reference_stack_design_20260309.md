# Public Reference Stack Design

## Goal

Extend nPathway beyond an MSigDB-only interpretation layer by supporting a broader, public pathway stack that can be used for:

- program annotation
- curated panel selection for multi-pathway enrichment views
- source-specific dashboard interpretation

## Principles

1. Dynamic programs remain primary.
2. Curated databases are interpretation layers, not the endpoint.
3. Public references should be grouped by source so users can tell whether a hit came from Reactome, WikiPathways, Pathway Commons, GO, Hallmark, or KEGG.
4. The default multi-pathway enrichment panel should be result-driven, not a fixed disease preset.

## Supported Public Sources

- Reactome
- WikiPathways
- Pathway Commons

These complement existing MSigDB collections:

- Hallmark
- GO BP
- KEGG

## Architecture

### 1. Download / cache layer

`src/npathway/data/public_references.py`

- resolves official download URLs
- caches files under `~/.npathway/data/public_references/`
- reads plain GMT, `.gmt.gz`, and zipped GMT archives

### 2. Harmonization layer

- prefixes each set with a stable source tag:
  - `REACTOME::`
  - `WP::`
  - `PATHWAYCOMMONS::`
- removes duplicate genes within a set
- optionally uppercases symbols
- enforces minimum gene-set size

### 3. Stack build layer

`scripts/build_public_reference_stack.py`

Produces:

- merged GMT
- JSON manifest with source-level counts

### 4. Annotation layer

`bulk_dynamic.py`

Now accepts:

- `reactome`
- `wikipathways`
- `pathwaycommons`

inside `annotation_collections`.

### 5. Dashboard layer

The dashboard now has to expose not just the strongest reference matches, but also the source each match came from. This prevents users from mixing GO, Hallmark, Reactome, and WikiPathways hits without noticing the knowledge layer difference.

## Multi-pathway panel policy

The canonical curated multi-pathway panel is now:

- result-driven
- limited to pathways present in the user-supplied curated GMT
- ranked using anchor-program overlap plus curated GSEA evidence

This replaced the older fixed AD-specific 5-pathway default.

## Recommended usage

### Broad public annotation

Use:

- `hallmark,go_bp,reactome,wikipathways,pathwaycommons`

when broad interpretation is desired.

### Tighter manuscript audit

Use:

- a project-specific curated GMT

when a manuscript needs a fixed reference layer with explicit provenance.

## Next extension

If the broader stack becomes the default for manuscript work, add:

- redundancy collapse across near-duplicate pathways
- source-aware panel selection
- optional regulatory layer support from OmniPath / regulon resources

