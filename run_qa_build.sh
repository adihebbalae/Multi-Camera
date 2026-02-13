#!/usr/bin/env bash
# General-purpose.
# Run dataset builder for all question categories. 
# Run from project root. Extra args (--limit, --output_dir, etc.) passed via "$@".

# NuScenes
# python nuscenes/datasetbuilder/nuscens_build.py --question_type counting "$@"
# python nuscenes/datasetbuilder/nuscens_build.py --question_type spatial "$@"
# python nuscenes/datasetbuilder/nuscens_build.py --question_type summarization "$@"

# Save videos for each scene.
python waymo/scenegraph/create_scenegraph_video.py \
  --dataroot /nas/standard_datasets/nuscenes \
  --scene-token 0c601ff2bf004fccafec366b08bf29e2 \
  --data-dir "/home/hg22723/projects/Multi-Camera/outputs" \
  --output-dir "/nas/neurosymbolic/multi-cam-dataset/nuscenes/videos_out" \
  --panoramic \
  --cameras CAM_FRONT_LEFT CAM_FRONT CAM_FRONT_RIGHT \

  # --no-annotations \
  # --cameras CAM_FRONT
  # --scene-token 0c601ff2bf004fccafec366b08bf29e2 \