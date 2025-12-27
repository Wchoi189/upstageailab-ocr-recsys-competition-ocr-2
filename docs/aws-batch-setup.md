# AWS Batch Cloud Processing Setup Guide

## Overview

This guide walks you through setting up AWS Batch + S3 + GitHub Actions for automated, cloud-based OCR pseudo-label generation using the Upstage API.

**Benefits**:
- ✅ Run processing remotely (no need to keep local machine on)
- ✅ Automatic scaling and cost optimization (spot instances)
- ✅ Resumable on failures (S3 checkpoints)
- ✅ CI/CD integration (auto-deploy on code changes)
- ✅ ~70% cost savings vs on-demand instances

---

## Prerequisites

- AWS Account with appropriate permissions
- GitHub repository with Actions enabled
- Upstage API key
- AWS CLI installed (completed ✅)

---

## Setup Steps

### 1. Configure AWS Credentials

First, configure your AWS CLI with your credentials:

```bash
aws configure
```

Enter:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., `us-east-1`)
- Default output format: `json`

### 2. Run AWS Infrastructure Setup

Execute the automated setup script:

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2

# Set your API key (will be stored in AWS Secrets Manager)
export UPSTAGE_API_KEY="your-api-key-here"

# Run setup
./scripts/cloud/aws-batch-setup.sh
```

This script creates:
- S3 bucket for data storage
- ECR repository for Docker images
- AWS Secrets Manager secret for API key
- IAM roles (Job, Execution, Service)
- AWS Batch compute environment (spot instances)
- AWS Batch job queue

**Output**: Configuration saved to `aws/config.env`

### 3. Upload Dataset to S3

```bash
# Source configuration
source aws/config.env

# Upload training dataset
aws s3 cp data/processed/baseline_train.parquet \
  s3://$S3_BUCKET/data/processed/baseline_train.parquet

# Upload validation dataset
aws s3 cp data/processed/baseline_val.parquet \
  s3://$S3_BUCKET/data/processed/baseline_val.parquet
```

### 4. Build and Push Docker Image

```bash
# Build Docker image
cd scripts/cloud
docker build -f Dockerfile.batch -t batch-processor:latest ../..

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ECR_URI

# Tag and push
docker tag batch-processor:latest $ECR_URI:latest
docker push $ECR_URI:latest
```

### 5. Configure GitHub Secrets

Add these secrets to your GitHub repository:

**Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key | IAM user credentials |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key | IAM user credentials |
| `AWS_REGION` | `us-east-1` (or your region) | AWS region |
| `ECR_REPOSITORY_URI` | From `aws/config.env` | Full ECR URI |
| `S3_BUCKET` | From `aws/config.env` | S3 bucket name |
| `JOB_ROLE_ARN` | From `aws/config.env` | IAM role ARN |
| `EXEC_ROLE_ARN` | From `aws/config.env` | Execution role ARN |
| `SECRET_ARN` | From `aws/config.env` | Secrets Manager ARN |

**Quick copy from config**:
```bash
source aws/config.env
echo "AWS_REGION: $AWS_REGION"
echo "ECR_REPOSITORY_URI: $ECR_URI"
echo "S3_BUCKET: $S3_BUCKET"
echo "JOB_ROLE_ARN: $JOB_ROLE_ARN"
echo "EXEC_ROLE_ARN: $EXEC_ROLE_ARN"
echo "SECRET_ARN: arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${SECRET_NAME}"
```

### 6. Register Batch Job Definition

Update the template with your values:

```bash
# Update placeholders in job definition
sed -i "s|<ECR_URI>|$ECR_URI|g" aws/batch-job-definition.json
sed -i "s|<JOB_ROLE_ARN>|$JOB_ROLE_ARN|g" aws/batch-job-definition.json
sed -i "s|<EXEC_ROLE_ARN>|$EXEC_ROLE_ARN|g" aws/batch-job-definition.json
sed -i "s|<SECRET_ARN>|arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${SECRET_NAME}|g" aws/batch-job-definition.json
sed -i "s|<S3_BUCKET>|$S3_BUCKET|g" aws/batch-job-definition.json

# Register job definition
aws batch register-job-definition --cli-input-json file://aws/batch-job-definition.json
```

---

## Usage

### Option A: GitHub Actions (Recommended)

1. **Auto-deployment**: Push code changes to `main` branch
   - Triggers: `.github/workflows/deploy-batch-processor.yml`
   - Automatically builds and pushes Docker image
   - Updates Batch job definition

2. **Run batch job**: Go to GitHub Actions
   - Select "Run Batch Processing Job"
   - Click "Run workflow"
   - Choose dataset: `baseline_train` or `baseline_val`
   - Configure parameters (resume, batch size, concurrency)
   - Click "Run workflow"

3. **Monitor progress**:
   - View workflow run in GitHub Actions
   - Check job status in AWS Batch console
   - Download results as artifact when complete

### Option B: AWS CLI (Manual)

```bash
# Submit job
aws batch submit-job \
  --job-name "pseudo-labels-baseline-train-$(date +%Y%m%d-%H%M%S)" \
  --job-definition pseudo-label-processor \
  --job-queue batch-processor-queue

# Check status
aws batch describe-jobs --jobs <JOB_ID>

# Download results
aws s3 cp s3://$S3_BUCKET/data/processed/baseline_train_pseudo_labels.parquet ./
```

---

## Monitoring & Troubleshooting

### View Logs

```bash
# Get job ID
JOB_ID=<your-job-id>

# Get log stream name
LOG_STREAM=$(aws batch describe-jobs --jobs $JOB_ID \
  --query 'jobs[0].container.logStreamName' --output text)

# View logs
aws logs tail /aws/batch/job --follow --log-stream-names $LOG_STREAM
```

### Check Checkpoints

```bash
# List checkpoints
aws s3 ls s3://$S3_BUCKET/checkpoints/ --recursive

# Download specific checkpoint
aws s3 cp s3://$S3_BUCKET/checkpoints/baseline_train_batch_0000.parquet ./
```

### Resume Failed Job

If a job fails mid-processing, resume from checkpoints:

```bash
aws batch submit-job \
  --job-name "pseudo-labels-resume-$(date +%Y%m%d-%H%M%S)" \
  --job-definition pseudo-label-processor \
  --job-queue batch-processor-queue \
  --container-overrides '{
    "command": [
      "python", "-m", "runners.batch_pseudo_labels_aws",
      "--dataset-name", "baseline_train",
      "--resume"
    ]
  }'
```

---

## Cost Estimation

For processing 3,272 images (~90 minutes):

- **Compute**: EC2 Spot (2 vCPU, 4GB) - ~$0.03
- **Storage**: S3 (temporary) - ~$0.01
- **Total**: **~$0.04** per full dataset

vs Local: $0 but requires machine to stay on for 90+ minutes

---

## Architecture Diagram

```
GitHub Repository
       ↓
   [Push to main]
       ↓
GitHub Actions (Deploy)
       ↓
   Docker Build
       ↓
Amazon ECR ──→ AWS Batch Job Definition
       ↓
GitHub Actions (Manual Trigger)
       ↓
AWS Batch Job Queue
       ↓
Fargate/EC2 Spot Instance
       ↓
[Process Images] ←→ Upstage API
       ↓
S3 (Checkpoints + Results)
       ↓
GitHub Actions (Download)
```

---

## Next Steps

After setup is complete:

1. **Test with small dataset**: Run validation set (404 images) first
2. **Monitor costs**: Check AWS Cost Explorer after first run
3. **Optimize**: Adjust batch size and concurrency based on results
4. **Scale**: Process full training set (3,272 images)

---

## Files Created

- `scripts/cloud/Dockerfile.batch` - Container definition
- `scripts/cloud/aws-batch-setup.sh` - Infrastructure setup
- `runners/batch_pseudo_labels_aws.py` - S3-enabled processor
- `.github/workflows/deploy-batch-processor.yml` - Auto-deployment
- `.github/workflows/run-batch-job.yml` - Manual job trigger
- `aws/batch-job-definition.json` - Batch job template
- `aws/config.env` - Configuration values

---

## Support

For issues:
1. Check CloudWatch logs (see Monitoring section)
2. Verify S3 bucket permissions
3. Ensure IAM roles have correct policies
4. Check GitHub Actions logs for deployment issues
