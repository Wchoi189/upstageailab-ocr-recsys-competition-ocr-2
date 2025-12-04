# Data Contract Organization Standards

## Structure

```
docs/pipeline/
├── README.md                    # Index and cross-references
├── data_contracts.md            # Core pipeline (dataset, transforms, model, loss)
├── inference-data-contracts.md  # Inference-specific (NEW)
├── preprocessing-data-contracts.md # Preprocessing module
└── DATA_CONTRACT_ORGANIZATION.md # This file
```

## Principles

1. **One contract per file** - Each major pipeline stage has its own contract file
2. **Ultra-concise** - AI-instruction focused, minimal verbosity, no tutorials
3. **Explicit cross-references** - Use file paths: `[Link](file.md#section)`
4. **Centralized index** - README.md provides navigation

## Contract File Template

```markdown
# [Contract Name]

**Purpose**: [One sentence]

## [Section Name]

[Concise definition - no tutorials]

**Fields**:
- `field`: Type - Description

**Validation Rules**:
- Rule 1
- Rule 2

## Related Contracts

- [Link](other-file.md#section)
```

## Cross-Reference Format

**Within same directory**:
```markdown
[Section Name](file.md#section-name)
```

**To other directories**:
```markdown
[Section Name](../path/to/file.md#section-name)
```

## Update Process

1. Update contract definition
2. Update README.md if structure changes
3. Update cross-references in related files
4. Update implementation
5. Update tests

## Missing Contracts

When identifying missing contracts:
1. Create new contract file following template
2. Add to README.md index
3. Add cross-references from related contracts
4. Update implementation to match contract
