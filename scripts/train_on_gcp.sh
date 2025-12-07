#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${PROJECT_ID:-"your-gcp-project"}
ZONE=${ZONE:-"us-central1-a"}
INSTANCE_NAME=${INSTANCE_NAME:-"wikiart-trainer"}
STARTUP_SCRIPT=${STARTUP_SCRIPT:-"./scripts/startup_train.sh"}

echo "Launching ${INSTANCE_NAME} in ${ZONE} (project=${PROJECT_ID})"

gcloud compute instances create "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="n1-standard-8" \
  --accelerator="type=nvidia-tesla-v100,count=1" \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB \
  --image-family="pytorch-latest-gpu" \
  --image-project="deeplearning-platform-release" \
  --metadata-from-file startup-script="${STARTUP_SCRIPT}"

echo "Instance ${INSTANCE_NAME} launched. Tail logs with:"
echo "gcloud compute ssh ${INSTANCE_NAME} --project ${PROJECT_ID} --zone ${ZONE}"
