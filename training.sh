gcloud ai-platform jobs submit training "modelCompileInside" \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region $REGION \
    --config aiplatformconfig.yaml