from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import open3d as o3d
from PIL import Image
import trimesh


DEFAULT_TARGET_TRIANGLES = 28000


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Generate a textured triangle LOD with Open3D quadric decimation and transfer "
          "the source UVs through closest-point barycentric interpolation."
      )
  )
  parser.add_argument("source", type=Path, help="Source textured OBJ")
  parser.add_argument("output", type=Path, help="Output LOD OBJ")
  target_group = parser.add_mutually_exclusive_group()
  target_group.add_argument(
      "--target-triangles",
      type=int,
      default=None,
      help=(
          "Target triangle count (default: 28000 when neither target option is specified)"
      ),
  )
  target_group.add_argument(
      "--triangle-ratio",
      type=float,
      default=None,
      help="Fraction of source triangles to retain, strictly between 0 and 1 (for example: 0.25)",
  )
  target_group.add_argument(
      "--normalize-only",
      action="store_true",
      help="Rewrite OBJ indices and triangulate faces without simplifying the geometry",
  )
  parser.add_argument(
      "--texture",
      type=Path,
      default=None,
      help="Source texture image (default: texture_map.png next to the source OBJ)",
  )
  return parser.parse_args()


def resolve_target_triangles(args: argparse.Namespace, source_triangles: int) -> int:
  if getattr(args, "normalize_only", False):
    return source_triangles
  if args.triangle_ratio is not None:
    if not np.isfinite(args.triangle_ratio) or not 0.0 < args.triangle_ratio < 1.0:
      raise RuntimeError("--triangle-ratio must be finite and strictly between 0 and 1")
    target_triangles = int(round(source_triangles * args.triangle_ratio))
    if target_triangles < 4:
      raise RuntimeError(
          f"--triangle-ratio {args.triangle_ratio} produces only {target_triangles} triangles; "
          "increase the ratio so the target is at least 4"
      )
  else:
    target_triangles = (
        args.target_triangles
        if args.target_triangles is not None
        else DEFAULT_TARGET_TRIANGLES
    )
    if target_triangles < 4:
      raise RuntimeError("--target-triangles must be at least 4")

  if target_triangles >= source_triangles:
    raise RuntimeError(
        f"Target {target_triangles} must be smaller than the source triangle count "
        f"{source_triangles}"
    )
  return target_triangles


def resolve_obj_index(raw_index: str, item_count: int, kind: str, line_number: int) -> int:
  try:
    index = int(raw_index)
  except ValueError as exc:
    raise RuntimeError(f"Invalid OBJ {kind} index {raw_index!r} at line {line_number}") from exc
  if index > 0:
    index -= 1
  elif index < 0:
    index += item_count
  else:
    raise RuntimeError(f"OBJ {kind} index cannot be zero at line {line_number}")
  if not 0 <= index < item_count:
    raise RuntimeError(
        f"OBJ {kind} index {raw_index} is out of range for {item_count} items "
        f"at line {line_number}"
    )
  return index


def load_obj_with_expanded_uvs(path: Path, texture_path: Path) -> trimesh.Trimesh:
  lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
  positions = []
  texture_coordinates = []
  for line_number, raw_line in enumerate(lines, start=1):
    fields = raw_line.split("#", 1)[0].split()
    if not fields:
      continue
    if fields[0] == "v":
      if len(fields) < 4:
        raise RuntimeError(f"OBJ vertex needs three coordinates at line {line_number}")
      positions.append(tuple(float(value) for value in fields[1:4]))
    elif fields[0] == "vt":
      if len(fields) < 3:
        raise RuntimeError(f"OBJ texture coordinate needs two values at line {line_number}")
      texture_coordinates.append(tuple(float(value) for value in fields[1:3]))

  if not positions:
    raise RuntimeError("OBJ does not contain any vertices")
  if not texture_coordinates:
    raise RuntimeError("OBJ does not contain any texture coordinates")

  expanded_vertices = []
  expanded_uvs = []
  triangles = []
  vertex_map = {}
  for line_number, raw_line in enumerate(lines, start=1):
    fields = raw_line.split("#", 1)[0].split()
    if not fields or fields[0] != "f":
      continue
    if len(fields) < 4:
      raise RuntimeError(f"OBJ face needs at least three vertices at line {line_number}")
    polygon = []
    for face_vertex in fields[1:]:
      indices = face_vertex.split("/")
      if len(indices) < 2 or not indices[0] or not indices[1]:
        raise RuntimeError(f"OBJ face is missing a UV index at line {line_number}")
      position_index = resolve_obj_index(indices[0], len(positions), "vertex", line_number)
      uv_index = resolve_obj_index(indices[1], len(texture_coordinates), "UV", line_number)
      key = (position_index, uv_index)
      expanded_index = vertex_map.get(key)
      if expanded_index is None:
        expanded_index = len(expanded_vertices)
        vertex_map[key] = expanded_index
        expanded_vertices.append(positions[position_index])
        expanded_uvs.append(texture_coordinates[uv_index])
      polygon.append(expanded_index)
    for index in range(1, len(polygon) - 1):
      triangles.append((polygon[0], polygon[index], polygon[index + 1]))

  if not triangles:
    raise RuntimeError("OBJ does not contain any faces")
  with Image.open(texture_path) as image:
    texture_image = image.convert("RGB")
  visual = trimesh.visual.texture.TextureVisuals(
      uv=np.asarray(expanded_uvs, dtype=np.float64),
      image=texture_image,
  )
  return trimesh.Trimesh(
      vertices=np.asarray(expanded_vertices, dtype=np.float64),
      faces=np.asarray(triangles, dtype=np.int64),
      visual=visual,
      process=False,
  )


def validate_textured_mesh(loaded: trimesh.Trimesh, path: Path) -> trimesh.Trimesh:
  if not isinstance(loaded, trimesh.Trimesh):
    raise RuntimeError(f"Expected one triangle mesh in {path}, got {type(loaded).__name__}")
  vertices = np.asarray(loaded.vertices)
  faces = np.asarray(loaded.faces)
  if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
    raise RuntimeError(f"Invalid source vertices: {vertices.shape}")
  if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
    raise RuntimeError(f"Invalid source triangles: {faces.shape}")
  if not isinstance(loaded.visual, trimesh.visual.texture.TextureVisuals):
    raise RuntimeError("Source mesh does not contain texture visuals")
  uv = np.asarray(loaded.visual.uv)
  if uv.shape != (len(vertices), 2) or not np.isfinite(uv).all():
    raise RuntimeError(f"Source UV array must be ({len(vertices)}, 2), got {uv.shape}")
  image = getattr(loaded.visual.material, "image", None)
  image_shape = None if image is None else np.asarray(image).shape
  if image_shape is None or len(image_shape) < 2 or min(image_shape[:2]) < 16:
    raise RuntimeError(
        f"Source texture was not loaded correctly (shape={image_shape}); check the OBJ mtllib reference"
    )
  return loaded


def require_textured_mesh(path: Path, texture_path: Path | None = None) -> trimesh.Trimesh:
  loaded = trimesh.load(path, force="mesh", process=False)
  try:
    return validate_textured_mesh(loaded, path)
  except RuntimeError as load_error:
    if texture_path is None:
      raise
    try:
      expanded = load_obj_with_expanded_uvs(path, texture_path)
      return validate_textured_mesh(expanded, path)
    except (OSError, RuntimeError, ValueError) as fallback_error:
      raise RuntimeError(
          f"Trimesh could not load usable textured geometry from {path}: {load_error}. "
          f"Direct OBJ UV loading also failed: {fallback_error}"
      ) from fallback_error


def to_open3d_mesh(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
  result = o3d.geometry.TriangleMesh()
  result.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
  result.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
  result.compute_vertex_normals()
  return result


def transfer_uvs(
    source_mesh: trimesh.Trimesh,
    lod_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  source_open3d = to_open3d_mesh(source_mesh)
  scene = o3d.t.geometry.RaycastingScene()
  scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(source_open3d))
  query = o3d.core.Tensor(np.asarray(lod_vertices, dtype=np.float32), dtype=o3d.core.Dtype.Float32)
  closest = scene.compute_closest_points(query)
  primitive_ids = closest["primitive_ids"].numpy().astype(np.int64)
  primitive_uvs = closest["primitive_uvs"].numpy().astype(np.float64)
  closest_points = closest["points"].numpy().astype(np.float64)
  if np.any(primitive_ids >= len(source_mesh.faces)):
    raise RuntimeError("Closest-point query returned an invalid source triangle index")

  source_faces = np.asarray(source_mesh.faces, dtype=np.int64)
  source_uv = np.asarray(source_mesh.visual.uv, dtype=np.float64)
  triangle_uv = source_uv[source_faces[primitive_ids]]
  barycentric = np.column_stack(
      (1.0 - primitive_uvs[:, 0] - primitive_uvs[:, 1], primitive_uvs[:, 0], primitive_uvs[:, 1])
  )
  lod_uv = np.einsum("ni,nij->nj", barycentric, triangle_uv)
  surface_distance = np.linalg.norm(lod_vertices - closest_points, axis=1)
  return lod_uv, surface_distance


def write_obj(
    output_path: Path,
    vertices: np.ndarray,
    normals: np.ndarray,
    uv: np.ndarray,
    faces: np.ndarray,
    texture_path: Path,
) -> tuple[Path, Path]:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  material_path = output_path.with_suffix(".mtl")
  destination_texture = output_path.parent / texture_path.name
  if texture_path.resolve() != destination_texture.resolve():
    shutil.copy2(texture_path, destination_texture)

  material_path.write_text(
      "\n".join(
          (
              "# FoundationPose textured render LOD material",
              "newmtl material_0",
              "Ns 250.000000",
              "Ka 1.000000 1.000000 1.000000",
              "Ks 0.500000 0.500000 0.500000",
              "Ke 0.000000 0.000000 0.000000",
              "Ni 1.500000",
              "d 1.000000",
              "illum 2",
              f"map_Kd {destination_texture.name}",
              "",
          )
      ),
      encoding="utf-8",
  )

  with output_path.open("w", encoding="utf-8") as file:
    file.write("# FoundationPose textured render LOD\n")
    file.write(f"mtllib {material_path.name}\n")
    file.write("o render_lod\n")
    for x, y, z in vertices:
      file.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
    for u, v in uv:
      file.write(f"vt {u:.9g} {v:.9g}\n")
    for nx, ny, nz in normals:
      file.write(f"vn {nx:.9g} {ny:.9g} {nz:.9g}\n")
    file.write("usemtl material_0\n")
    for face in faces:
      indices = face.astype(np.int64) + 1
      file.write("f " + " ".join(f"{index}/{index}/{index}" for index in indices) + "\n")
  return material_path, destination_texture


def main() -> int:
  args = parse_args()
  source_path = args.source.expanduser().resolve()
  output_path = args.output.expanduser().resolve()
  texture_path = (
      args.texture.expanduser().resolve()
      if args.texture is not None
      else source_path.with_name("texture_map.png")
  )
  if not source_path.is_file():
    raise RuntimeError(f"Source OBJ does not exist: {source_path}")
  if not texture_path.is_file():
    raise RuntimeError(f"Texture image does not exist: {texture_path}")

  source_mesh = require_textured_mesh(source_path, texture_path)
  source_triangles = int(len(source_mesh.faces))
  target_triangles = resolve_target_triangles(args, source_triangles)

  if args.normalize_only:
    lod_vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    lod_faces = np.asarray(source_mesh.faces, dtype=np.int64)
    lod_normals = np.asarray(source_mesh.vertex_normals, dtype=np.float64)
    lod_uv = np.asarray(source_mesh.visual.uv, dtype=np.float64)
    surface_distance = np.zeros(len(lod_vertices), dtype=np.float64)
  else:
    lod_open3d = to_open3d_mesh(source_mesh).simplify_quadric_decimation(target_triangles)
    lod_open3d.remove_degenerate_triangles()
    lod_open3d.remove_duplicated_triangles()
    lod_open3d.remove_unreferenced_vertices()
    lod_open3d.compute_vertex_normals()
    lod_vertices = np.asarray(lod_open3d.vertices, dtype=np.float64)
    lod_faces = np.asarray(lod_open3d.triangles, dtype=np.int64)
    lod_normals = np.asarray(lod_open3d.vertex_normals, dtype=np.float64)
    if (
        len(lod_faces) == 0
        or not np.isfinite(lod_vertices).all()
        or not np.isfinite(lod_normals).all()
    ):
      raise RuntimeError("Quadric decimation produced an invalid mesh")
    lod_uv, surface_distance = transfer_uvs(source_mesh, lod_vertices)

  if lod_uv.shape != (len(lod_vertices), 2) or not np.isfinite(lod_uv).all():
    raise RuntimeError(f"Output UV array must be ({len(lod_vertices)}, 2), got {lod_uv.shape}")
  material_path, destination_texture = write_obj(
      output_path,
      lod_vertices,
      lod_normals,
      lod_uv,
      lod_faces,
      texture_path,
  )

  validated = require_textured_mesh(output_path, destination_texture)
  source_bounds = np.asarray(source_mesh.bounds, dtype=np.float64)
  lod_bounds = np.asarray(validated.bounds, dtype=np.float64)
  source_diagonal = max(float(np.linalg.norm(source_bounds[1] - source_bounds[0])), 1e-12)
  summary = {
      "source_obj": str(source_path),
      "output_obj": str(output_path),
      "material": str(material_path),
      "texture": str(destination_texture),
      "source_vertices": int(len(source_mesh.vertices)),
      "source_triangles": source_triangles,
      "operation": "normalize" if args.normalize_only else "decimate",
      "target_triangles": target_triangles,
      "requested_triangle_ratio": args.triangle_ratio,
      "lod_vertices": int(len(validated.vertices)),
      "lod_triangles": int(len(validated.faces)),
      "triangle_ratio": float(len(validated.faces) / source_triangles),
      "mean_vertex_to_source_surface": float(surface_distance.mean()),
      "max_vertex_to_source_surface": float(surface_distance.max()),
      "mean_vertex_to_source_surface_ratio": float(surface_distance.mean() / source_diagonal),
      "max_vertex_to_source_surface_ratio": float(surface_distance.max() / source_diagonal),
      "bounds_center_delta_ratio": float(
          np.linalg.norm(lod_bounds.mean(axis=0) - source_bounds.mean(axis=0)) / source_diagonal
      ),
      "bounds_extent_delta_ratio": float(
          np.max(np.abs(np.diff(lod_bounds, axis=0) - np.diff(source_bounds, axis=0))) / source_diagonal
      ),
      "texture_shape": list(np.asarray(validated.visual.material.image).shape),
  }
  summary_path = output_path.with_suffix(".json")
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
