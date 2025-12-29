# AWS Batch Quick Reference

## 🚀 Quick Start Commands

### 1. Initial Setup
```bash
# Configure AWS CLI
aws configure

# Set API key and run infrastructure setup
export UPSTAGE_API_KEY="your-key"
./scripts/cloud/aws-batch-setup.sh

# Upload datasets
source aws/config.env
aws s3 cp data/processed/baseline_train.parquet s3://$S3_BUCKET/data/processed/
aws s3 cp data/processed/baseline_val.parquet s3://$S3_BUCKET/data/processed/
```

### 2. Build & Deploy
```bash
# Build Docker image
cd scripts/cloud
docker build -f Dockerfile.batch -t batch-processor ../..

# Push to ECR
source ../../aws/config.env
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ECR_URI
docker tag batch-processor:latest $ECR_URI:latest
docker push $ECR_URI:latest

# Register job definition
cd ../..
aws batch register-job-definition --cli-input-json file://aws/batch-job-definition.json
```

### 3. Run Job
```bash
# Via AWS CLI
aws batch submit-job \
  --job-name "pseudo-labels-$(date +%Y%m%d-%H%M%S)" \
  --job-definition pseudo-label-processor \
  --job-queue batch-processor-queue

# Via GitHub Actions (preferred)
# Go to: Actions → Run Batch Processing Job → Run workflow
```

---

## 📋 GitHub Secrets Checklist

After running setup, add these to GitHub:

```bash
# Display values for GitHub secrets
source aws/config.env
echo "AWS_REGION: $AWS_REGION"
echo "ECR_REPOSITORY_URI: $ECR_URI"
echo "S3_BUCKET: $S3_BUCKET"
echo "JOB_ROLE_ARN: $JOB_ROLE_ARN"
echo "EXEC_ROLE_ARN: $EXEC_ROLE_ARN"
echo "SECRET_ARN: arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${SECRET_NAME}"
```

Required secrets:
- [ ] AWS_ACCESS_KEY_ID
- [ ] AWS_SECRET_ACCESS_KEY
- [ ] AWS_REGION
- [ ] ECR_REPOSITORY_URI
- [ ] S3_BUCKET
- [ ] JOB_ROLE_ARN
- [ ] EXEC_ROLE_ARN
- [ ] SECRET_ARN

---

## 🔍 Monitoring Commands

```bash
# List running jobs
aws batch list-jobs --job-queue batch-processor-queue --job-status RUNNING

# Check job status
aws batch describe-jobs --jobs <JOB_ID>

# View logs
JOB_ID=<your-job-id>
LOG_STREAM=$(aws batch describe-jobs --jobs $JOB_ID --query 'jobs[0].container.logStreamName' --output text)
aws logs tail /aws/batch/job --follow --log-stream-names $LOG_STREAM

# List checkpoints
aws s3 ls s3://$S3_BUCKET/checkpoints/ --recursive

# Download results
aws s3 cp s3://$S3_BUCKET/data/processed/baseline_train_pseudo_labels.parquet ./
```

---

## 💰 Cost Optimization

- **Use Spot Instances**: Already configured (70% savings)
- **Right-size resources**: 2 vCPU, 4GB is optimal for this workload
- **Clean up**: Delete datasets from S3 after downloading
- **Monitor**: Check AWS Cost Explorer weekly

**Expected cost**: ~$0.09 per full dataset run (3,272 images)

---

## 🐛 Troubleshooting

### Job Stuck in RUNNABLE
```bash
# Check compute environment
aws batch describe-compute-environments --compute-environments batch-processor-compute-env
```

### Job Failed
```bash
# Get failure reason
aws batch describe-jobs --jobs <JOB_ID> --query 'jobs[0].statusReason'
```

### Resume Failed Job
```bash
aws batch submit-job \
  --job-name "pseudo-labels-resume" \
  --job-definition pseudo-label-processor \
  --job-queue batch-processor-queue \
  --container-overrides '{"command": ["python", "-m", "runners.batch_pseudo_labels_aws", "--resume"]}'
```

---

## 📁 File Locations

- Setup script: `scripts/cloud/aws-batch-setup.sh`
- Dockerfile: `scripts/cloud/Dockerfile.batch`
- Processor: `runners/batch_pseudo_labels_aws.py`
- Job definition: `aws/batch-job-definition.json`
- Config: `aws/config.env` (generated)
- Workflows: `.github/workflows/deploy-batch-processor.yml`, `run-batch-job.yml`

---

## 🔗 Useful Links

- [Full Setup Guide](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/aws-batch-setup.md)
- [GitHub Secrets Guide](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/github-secrets-setup.md)
- [AWS Batch Console](https://console.aws.amazon.com/batch/)
- [S3 Console](https://console.aws.amazon.com/s3/)
- [ECR Console](https://console.aws.amazon.com/ecr/)
