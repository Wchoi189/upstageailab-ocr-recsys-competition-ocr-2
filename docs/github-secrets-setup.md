# GitHub Secrets Configuration

## Required Secrets

Configure these in your GitHub repository:

**Settings → Secrets and variables → Actions → New repository secret**

### Authentication Secrets

| Secret Name | How to Get | Example |
|-------------|-----------|---------|
| `AWS_ACCESS_KEY_ID` | AWS IAM Console → Users → Security credentials | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | Created with access key (shown once) | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |

### AWS Resource Secrets

After running `scripts/cloud/aws-batch-setup.sh`, get these from `aws/config.env`:

```bash
# Source the config file
source aws/config.env

# Display all secret values
echo "AWS_REGION=$AWS_REGION"
echo "ECR_REPOSITORY_URI=$ECR_URI"
echo "S3_BUCKET=$S3_BUCKET"
echo "JOB_ROLE_ARN=$JOB_ROLE_ARN"
echo "EXEC_ROLE_ARN=$EXEC_ROLE_ARN"
echo "SECRET_ARN=arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${SECRET_NAME}"
```

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `AWS_REGION` | AWS region for resources | `us-east-1` |
| `ECR_REPOSITORY_URI` | Full ECR repository URI | `123456789012.dkr.ecr.us-east-1.amazonaws.com/batch-processor` |
| `S3_BUCKET` | S3 bucket for data | `ocr-batch-processing` |
| `JOB_ROLE_ARN` | IAM role for Batch jobs | `arn:aws:iam::123456789012:role/BatchProcessorJobRole` |
| `EXEC_ROLE_ARN` | Task execution role | `arn:aws:iam::123456789012:role/BatchProcessorExecutionRole` |
| `SECRET_ARN` | Secrets Manager ARN | `arn:aws:secretsmanager:us-east-1:123456789012:secret:upstage-api-key` |

## Quick Setup Script

Copy and run this after AWS infrastructure is set up:

```bash
#!/bin/bash
# Get values from AWS config
source aws/config.env

echo "==========================================
GitHub Secrets Configuration
=========================================="
echo ""
echo "Add these secrets to your GitHub repository:"
echo "Settings → Secrets and variables → Actions"
echo ""
echo "1. AWS_ACCESS_KEY_ID"
echo "   Value: [from AWS IAM Console]"
echo ""
echo "2. AWS_SECRET_ACCESS_KEY"
echo "   Value: [from AWS IAM Console]"
echo ""
echo "3. AWS_REGION"
echo "   Value: $AWS_REGION"
echo ""
echo "4. ECR_REPOSITORY_URI"
echo "   Value: $ECR_URI"
echo ""
echo "5. S3_BUCKET"
echo "   Value: $S3_BUCKET"
echo ""
echo "6. JOB_ROLE_ARN"
echo "   Value: $JOB_ROLE_ARN"
echo ""
echo "7. EXEC_ROLE_ARN"
echo "   Value: $EXEC_ROLE_ARN"
echo ""
echo "8. SECRET_ARN"
echo "   Value: arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${SECRET_NAME}"
echo ""
echo "=========================================="
```

## Verification

After adding secrets, verify they're set correctly:

1. Go to your repository on GitHub
2. Navigate to **Settings → Secrets and variables → Actions**
3. You should see all 8 secrets listed
4. Click **"Update"** on any secret to view (but not reveal) its value

## Testing

To test if secrets are working:

1. Go to **Actions** tab
2. Select **"Deploy Batch Processor"** workflow
3. Click **"Run workflow"**
4. If it succeeds, all secrets are configured correctly

## Troubleshooting

### Secret Not Found
**Error**: `Secret AWS_REGION is not set`

**Fix**: Ensure secret name matches exactly (case-sensitive)

### Invalid ARN
**Error**: `Invalid ARN format`

**Fix**: ARN should start with `arn:aws:` and contain no quotes

### ECR Login Failed
**Error**: `error getting credentials`

**Fix**: Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are correct

### Permission Denied
**Error**: `User: arn:aws:iam::... is not authorized`

**Fix**: IAM user needs permissions for ECR, Batch, and S3
