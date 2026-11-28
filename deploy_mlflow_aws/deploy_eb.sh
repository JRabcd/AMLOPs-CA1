#!/usr/bin/env bash
# Deploy or redeploy the team MLflow server on AWS Elastic Beanstalk in one command (PBI-14.1).
#
# Run it in AWS CloudShell (Learner Lab console, bottom-left "CloudShell"), in a folder that holds the
# deployment zip (Actions -> Upload file):
#
#     bash deploy_eb.sh                 # redeploy mlflow-eb-v3.zip to the existing environment
#     bash deploy_eb.sh other.zip       # redeploy another bundle
#
# If the environment doesn't exist yet, the same command creates the S3 bucket, the application and the
# environment (single t3.medium instance, LabRole / LabInstanceProfile) and asks for the RDS password, which
# is stored only as an Elastic Beanstalk environment property. RDS must already allow MLflow's triggers
# (parameter group with log_bin_trust_function_creators = 1, see docs/it-manager-guide.md section 5).
#
# Settings can be changed with environment variables: APP, ENV_NAME, CNAME, BUCKET, DB_HOST, DB_USER,
# INSTANCE_TYPE.
set -euo pipefail

ZIP=${1:-mlflow-eb-v3.zip}
APP=${APP:-spbuy-mlflow}
ENV_NAME=${ENV_NAME:-spbuy-mlflow-env}
CNAME=${CNAME:-spbuy-mlflow-group3}
DB_HOST=${DB_HOST:-spbuy-db.cpgyysqoa2uu.us-east-1.rds.amazonaws.com}
DB_USER=${DB_USER:-admin}
INSTANCE_TYPE=${INSTANCE_TYPE:-t3.medium}   # t3.small (2 GB) ran out of memory
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=${BUCKET:-spbuy-mlflow-$ACCOUNT}
VERSION=v$(date -u +%Y%m%d-%H%M%S)

[ -f "$ZIP" ] || { echo "Deployment zip '$ZIP' not found in $(pwd). Upload it first (Actions -> Upload file)."; exit 1; }

echo "== Bucket $BUCKET (model files and deployment bundles)"
aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1 || aws s3 mb "s3://$BUCKET"
aws s3 cp "$ZIP" "s3://$BUCKET/deploy/$VERSION.zip" --only-show-errors

echo "== Application $APP, version $VERSION"
if [ "$(aws elasticbeanstalk describe-applications --application-names "$APP" --query 'length(Applications)' --output text)" = "0" ]; then
  aws elasticbeanstalk create-application --application-name "$APP" --description "Team MLflow server" >/dev/null
fi
aws elasticbeanstalk create-application-version --application-name "$APP" --version-label "$VERSION" \
  --source-bundle "S3Bucket=$BUCKET,S3Key=deploy/$VERSION.zip" >/dev/null

STATUS=$(aws elasticbeanstalk describe-environments --application-name "$APP" --environment-names "$ENV_NAME" \
  --query 'Environments[?Status!=`Terminated`]|[0].Status' --output text)

if [ "$STATUS" != "None" ]; then
  echo "== Redeploying $ENV_NAME (currently $STATUS)"
  aws elasticbeanstalk wait environment-updated --application-name "$APP" --environment-names "$ENV_NAME"
  aws elasticbeanstalk update-environment --environment-name "$ENV_NAME" --version-label "$VERSION" >/dev/null
else
  echo "== Creating $ENV_NAME"
  if [ -z "${DB_PASSWORD:-}" ]; then read -rsp "RDS password for $DB_USER (not shown): " DB_PASSWORD; echo; fi
  STACK=$(aws elasticbeanstalk list-available-solution-stacks \
    --query "SolutionStacks[?contains(@, 'Amazon Linux 2023') && contains(@, 'running Docker')]|[0]" --output text)
  ENV_OPT=aws:elasticbeanstalk:application:environment
  aws elasticbeanstalk create-environment --application-name "$APP" --environment-name "$ENV_NAME" \
    --cname-prefix "$CNAME" --solution-stack-name "$STACK" --version-label "$VERSION" --option-settings \
    Namespace=aws:autoscaling:launchconfiguration,OptionName=IamInstanceProfile,Value=LabInstanceProfile \
    Namespace=aws:elasticbeanstalk:environment,OptionName=ServiceRole,Value=LabRole \
    Namespace=aws:elasticbeanstalk:environment,OptionName=EnvironmentType,Value=SingleInstance \
    Namespace=aws:ec2:instances,OptionName=InstanceTypes,Value="$INSTANCE_TYPE" \
    Namespace=$ENV_OPT,OptionName=DB_HOST,Value="$DB_HOST" \
    Namespace=$ENV_OPT,OptionName=DB_USER,Value="$DB_USER" \
    Namespace=$ENV_OPT,OptionName=DB_PASSWORD,Value="$DB_PASSWORD" \
    Namespace=$ENV_OPT,OptionName=MLFLOW_S3_BUCKET,Value="$BUCKET" \
    Namespace=$ENV_OPT,OptionName=MLFLOW_WORKERS,Value=1 >/dev/null
fi

echo "== Waiting for the environment (about 3-6 minutes)"
aws elasticbeanstalk wait environment-updated --application-name "$APP" --environment-names "$ENV_NAME"
read -r HEALTH URL < <(aws elasticbeanstalk describe-environments --environment-names "$ENV_NAME" \
  --query 'Environments[0].[Health,CNAME]' --output text)
for _ in $(seq 1 30); do
  curl -fs -m 10 "http://$URL/health" >/dev/null && break
  sleep 10
done
if curl -fs -m 10 "http://$URL/health" >/dev/null; then
  echo "== Done: http://$URL is up (health $HEALTH, version $VERSION). Put it in .env as MLFLOW_TRACKING_URI."
else
  echo "== Deployed version $VERSION, but http://$URL is not answering yet (health $HEALTH)."
  echo "   Check Elastic Beanstalk -> $ENV_NAME -> Logs, or the troubleshooting notes in docs/it-manager-guide.md."
  exit 1
fi
