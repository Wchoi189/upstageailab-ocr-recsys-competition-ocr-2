# KIE Implementation Tasks

## Priority Issues

- [x] Fix UTF-8 encoding for Korean text in visualization pipeline
  - [x] Locate visualization generation code
  - [x] Add UTF-8 encoding specification
  - [x] Test on Korean text samples
- [x] Validate and adjust sepia enhancement parameters
  - [x] Review current sepia implementation
  - [x] Identify brightness issue
  - [x] Test adjusted parameters
- [/] Run enhanced API experiment with logging
  - [x] Add API response logging to pseudo-label script
  - [ ] Test on sample batch (10-20 images)
  - [ ] Validate coordinate mapping accuracy
- [x] Implement API usage tracker
  - [x] Create tracker module
  - [x] Add usage logging to API calls
  - [x] Create usage report command

## KIE Pseudo-Labeling Pipeline

- [ ] Sample batch validation (50 images)
  - [ ] Extract diverse samples from baseline dataset
  - [ ] Generate pseudo-labels with/without enhancement
  - [ ] Compare results
  - [ ] Create validation report
- [ ] Full dataset pseudo-labeling
  - [ ] Implement batch processing with checkpointing
  - [ ] Process baseline_train (4,089 images)
  - [ ] Process baseline_val (404 images)
  - [ ] Quality assurance on random samples

## Long-Term Tasks

- [ ] KIE model selection and implementation
- [ ] Text recognition dataset acquisition strategy
