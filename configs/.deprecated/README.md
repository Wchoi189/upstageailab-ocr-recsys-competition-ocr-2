# Deprecated Configurations

These configurations have been archived and are no longer used in the main training pipeline.

## schemas/ (4+ files)
- **Reason**: Appear to be snapshot copies of config structure, not referenced anywhere
- **Benefit of removal**: Eliminates confusion about which schema is authoritative
- **Recovery**: All configs still functional without these files

## benchmark/ (1 file)
- **Reason**: Unused in current training/evaluation pipeline
- **Benefit of removal**: Clarifies what configs are actively used
- **Recovery**: Can be restored if needed for benchmarking work

## tools/ (empty or misc)
- **Reason**: Placeholder directory, no configs
- **Benefit of removal**: Reduces clutter for new users

## Restoring Archived Configs

If you need a config from this directory:
1. Copy it back to the appropriate location in configs/
2. Update train.yaml defaults to include it
3. Test with a training run
