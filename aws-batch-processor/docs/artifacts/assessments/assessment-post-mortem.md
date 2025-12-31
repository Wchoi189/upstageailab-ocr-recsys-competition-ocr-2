
================================================================================
POST-MORTEM: Prebuilt Extraction Batch Job Failures
================================================================================

WHAT WENT WRONG:
----------------
1. Checkpointing was too infrequent (every 500 images)
   - baseline_val (404 images): Never checkpointed
   - baseline_test (413 images): Never checkpointed  
   - baseline_train (3,272 images): Only checkpointed at 500-image intervals

2. Results accumulated in memory, lost on timeout
   - 46 images processed in baseline_test but lost
   - No way to recover partial results

3. Jobs timed out after 6 hours
   - Even with concurrency=1, processing too slow
   - Rate limiting from Prebuilt Extraction API
   - No checkpoints to resume from

4. --resume flag existed but couldn't help
   - No checkpoints were created for small datasets
   - Nothing to resume from

ROOT CAUSE:
-----------
The checkpointing design assumed batch_size=500 would always be reached.
For datasets < 500 images, checkpoints were never saved until final output.
Final output only saved at the very end, so timeouts = total loss.

WHAT'S FIXED:
-------------
1. Checkpointing now saves every 100 images (or batch_size if smaller)
   - Ensures checkpoints even for small datasets
   - Better recovery from timeouts

2. Checkpoints save cumulative results
   - Can resume from any checkpoint
   - No data loss between checkpoints

3. More frequent saves = better fault tolerance
   - Even if job times out, can resume from last checkpoint
   - Maximum loss: 100 images instead of entire dataset

LESSONS LEARNED:
---------------
- For long-running batch jobs, checkpoint frequently (every 50-100 items)
- Don't assume datasets will always be large enough to hit batch_size
- Test checkpoint/resume functionality with small datasets
- Consider API rate limits when estimating processing time

================================================================================
