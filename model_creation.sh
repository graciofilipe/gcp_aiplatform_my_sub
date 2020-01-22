gcloud ai-platform versions create $VERSION_NAME \
  --model $MODEL_NAME \
  --origin $MODEL_DIR \
  --runtime-version=1.15 \
  --framework $FRAMEWORK \
  --python-version=3.7