#!/bin/bash
set -e

# Configuración
DATASET_PATH="/root/devops-ai/datasets/ai-nerf-datasets/lego-ds"
IMG_DIR="${DATASET_PATH}/images"
COLMAP_DIR="${DATASET_PATH}/colmap"
SPARSE_DIR="${COLMAP_DIR}/sparse"
TEXT_DIR="${SPARSE_DIR}/0_text"
DB_PATH="${COLMAP_DIR}/database.db"
TRANSFORMS_PATH="${DATASET_PATH}/transforms.json"

# Limpieza
echo "==> Limpiando anteriores"
rm -rf "$COLMAP_DIR"
rm -f "$TRANSFORMS_PATH"

mkdir -p "$COLMAP_DIR"

# 1. Extracción de características
echo "==> Extrayendo características"
colmap feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$IMG_DIR" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV

# 2. Matching secuencial
echo "==> Realizando matching secuencial"
colmap sequential_matcher \
  --database_path "$DB_PATH" \
  --SiftMatching.use_gpu 1

# 3. Matching exhaustivo adicional (complementario)
echo "==> Realizando matching exhaustivo complementario"
colmap exhaustive_matcher \
  --database_path "$DB_PATH"

# 4. Mapeo
echo "==> Reconstruyendo modelo (mapper)"
mkdir -p "$SPARSE_DIR"
colmap mapper \
  --database_path "$DB_PATH" \
  --image_path "$IMG_DIR" \
  --output_path "$SPARSE_DIR" \
  --Mapper.ba_global_max_refinements 5 \
  --Mapper.min_num_matches 5 \
  --Mapper.init_min_tri_angle 1

# 5. Conversión a formato TXT
echo "==> Convirtiendo modelo a formato TXT"
mkdir -p "$TEXT_DIR"
colmap model_converter \
  --input_path "$SPARSE_DIR/0" \
  --output_path "$TEXT_DIR" \
  --output_type TXT

# 6. Generación de transforms.json
echo "==> Generando transforms.json para Instant-NGP"
python3 colmap2nerf.py \
  --images "$IMG_DIR" \
  --text "$TEXT_DIR" \
  --colmap_db "$DB_PATH" \
  --out "$TRANSFORMS_PATH" \
  --colmap_camera_model OPENCV \
  --aabb_scale 2

# 7. Validación
if [ -f "$TRANSFORMS_PATH" ]; then
  echo "✅ transforms.json generado correctamente"
  echo "🔎 Primeras líneas:"
  head -n 20 "$TRANSFORMS_PATH"
else
  echo "❌ Error: transforms.json no generado"
  exit 1
fi
