#!/usr/bin/env python3
"""
Build SWE-bench Pro Docker images locally and tag them like the prebuilts
(jefzda/sweap-images:<dockerhub_tag>).

Reads instance_id -> dockerhub_tag from data/instances.jsonl, and uses
Dockerfiles from SWE-bench_Pro-os/dockerfiles/ (base then instance).
For each instance:
  1. Tries to pull the base image (from instance Dockerfile's FROM).
  2. If pull fails (e.g. ECR not accessible or short name like base_NodeBB__NodeBB
     rejected by Docker Hub), builds the base from base_dockerfile with ECR
     FROM lines replaced by Docker Hub equivalents, and tags it as
     sweap-local/<lowercase-base-name>.
  3. Builds the instance image (using a temp Dockerfile with FROM set to the
     local base tag when applicable) and tags it as <dockerhub_username>/sweap-images:<dockerhub_tag>.

Resulting images can be used by the run step without code changes: use the same
dockerhub_username (default jefzda) and Docker will use your local images.

Usage:
  uv run python scripts/build_sweap_images.py --limit 2
  uv run python scripts/build_sweap_images.py --instance-ids instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan
  uv run python scripts/build_sweap_images.py --dockerhub-username jefzda --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # nosec B404
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    assert True, "argparse entry"
    parser = argparse.ArgumentParser(
        description="Build sweap-images locally and tag as jefzda/sweap-images:<tag>"
    )
    parser.add_argument(
        "--swe-bench-pro-dir",
        type=Path,
        default=Path("SWE-bench_Pro-os"),
        help="Path to SWE-bench_Pro-os repo (contains dockerfiles/)",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        default=Path("data/instances.jsonl"),
        help="JSONL of instances with instance_id and dockerhub_tag",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Build only the first N instances"
    )
    parser.add_argument("--instance-ids", type=str, nargs="*", default=None)
    parser.add_argument("--dockerhub-username", type=str, default="jefzda")
    parser.add_argument("--build-base-if-pull-fails", action="store_true", default=True)
    parser.add_argument(
        "--no-build-base-if-pull-fails",
        action="store_false",
        dest="build_base_if_pull_fails",
    )
    result = parser.parse_args()
    assert result is not None, "parsed args must not be None"
    return result


def _resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path] | None:
    assert args is not None, "args must not be None"
    assert hasattr(args, "swe_bench_pro_dir"), "args must have swe_bench_pro_dir"
    repo_root = Path(__file__).resolve().parent.parent
    swe_dir = (repo_root / args.swe_bench_pro_dir).resolve()
    if not swe_dir.is_dir():
        print(f"Error: SWE-bench Pro dir not found: {swe_dir}", file=sys.stderr)
        return None
    base_dir = swe_dir / "dockerfiles" / "base_dockerfile"
    instance_dir = swe_dir / "dockerfiles" / "instance_dockerfile"
    if not base_dir.is_dir() or not instance_dir.is_dir():
        print(f"Error: dockerfiles not found under {swe_dir}", file=sys.stderr)
        return None
    return swe_dir, base_dir, instance_dir


def _load_id_to_tag(instances_file: Path) -> dict[str, str] | None:
    assert instances_file is not None, "instances_file must not be None"
    assert isinstance(instances_file, Path), "instances_file must be a Path"
    if not instances_file.exists():
        print(f"Error: instances file not found: {instances_file}", file=sys.stderr)
        return None
    id_to_tag: dict[str, str] = {}
    with instances_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            iid = obj.get("instance_id")
            tag = obj.get("dockerhub_tag")
            if iid and tag and iid not in id_to_tag:
                id_to_tag[iid] = tag
    return id_to_tag


def _filter_to_build(id_to_tag: dict[str, str], args: argparse.Namespace) -> list[str]:
    assert id_to_tag is not None, "id_to_tag must not be None"
    assert args is not None, "args must not be None"
    to_build = list(id_to_tag.keys())
    if args.instance_ids is not None:
        to_build = [i for i in to_build if i in args.instance_ids]
        if len(to_build) != len(args.instance_ids):
            missing = set(args.instance_ids) - set(to_build)
            print(
                f"Warning: instances not in instances file: {missing}", file=sys.stderr
            )
    if args.limit is not None:
        to_build = to_build[: args.limit]
    return to_build


def local_base_tag(base_name: str) -> str:
    assert base_name is not None, "base_name must not be None"
    assert isinstance(base_name, str), "base_name must be a string"
    return "sweap-local/" + base_name.lower()


def _read_base_image(dockerfile_path: Path) -> str | None:
    assert dockerfile_path is not None, "dockerfile_path must not be None"
    assert dockerfile_path.exists(), f"Dockerfile must exist: {dockerfile_path}"
    with dockerfile_path.open() as f:
        for line in f:
            if line.strip().startswith("FROM "):
                return line.strip().split(maxsplit=1)[1].strip()
    return None


def _build_base_locally(
    instance_id: str, base_image: str, swe_dir: Path, base_dir: Path
) -> str | None:
    assert instance_id, "instance_id must not be empty"
    assert base_image, "base_image must not be empty"
    base_dockerfile_path = base_dir / instance_id / "Dockerfile"
    if not base_dockerfile_path.exists():
        return None
    content = base_dockerfile_path.read_text()
    new_content = re.sub(
        r"084828598639\.dkr\.ecr\.us-west-2\.amazonaws\.com/docker-hub/library/",
        "docker.io/library/",
        content,
    )
    if new_content == content:
        new_content = re.sub(
            r"084828598639\.dkr\.ecr\.us-west-2\.amazonaws\.com/", "", content, count=1
        )
    local_tag = local_base_tag(base_image)
    tmp = swe_dir / "dockerfiles" / ".build_base.Dockerfile"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(new_content)
    try:
        ok = run_docker(
            "build",
            "-f",
            str(tmp.relative_to(swe_dir)),
            "-t",
            local_tag,
            ".",
            cwd=swe_dir,
        )
        if not ok:
            print(f"Failed to build base for {instance_id}")
            return None
        print(f"Built base and tagged as {local_tag}")
        return local_tag
    finally:
        tmp.unlink(missing_ok=True)


def _get_from_image(
    instance_id: str, base_image: str, swe_dir: Path, base_dir: Path, build_base: bool
) -> str | None:
    assert instance_id, "instance_id must not be empty"
    assert base_image, "base_image must not be empty"
    pull_ok = run_docker("pull", base_image, cwd=swe_dir)
    if pull_ok:
        return base_image
    if not build_base:
        print(
            f"Skip {instance_id}: base image pull failed and --no-build-base-if-pull-fails"
        )
        return None
    print(f"Building base image for {instance_id}...")
    result = _build_base_locally(instance_id, base_image, swe_dir, base_dir)
    if result is None:
        print(
            f"Skip {instance_id}: base image pull failed and no base Dockerfile found"
        )
    return result


def _build_instance_image(
    instance_id: str,
    final_image: str,
    from_image: str,
    base_image: str,
    instance_df: Path,
    swe_dir: Path,
) -> None:
    assert instance_id, "instance_id must not be empty"
    assert final_image, "final_image must not be empty"
    df_to_use = instance_df
    tmp = None
    if from_image != base_image:
        content = instance_df.read_text()
        new_from = re.sub(r"^FROM\s+\S+", f"FROM {from_image}", content, count=1)
        tmp = swe_dir / "dockerfiles" / ".build_instance.Dockerfile"
        tmp.write_text(new_from)
        df_to_use = tmp
    try:
        print(f"Building {final_image} ...")
        if not run_docker(
            "build",
            "-f",
            str(df_to_use.relative_to(swe_dir)),
            "-t",
            final_image,
            ".",
            cwd=swe_dir,
        ):
            print(f"Failed to build {final_image}")
            return
        print(f"Built and tagged {final_image}")
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink(missing_ok=True)


def _build_one(
    instance_id: str,
    id_to_tag: dict[str, str],
    swe_dir: Path,
    base_dir: Path,
    instance_dir: Path,
    args: argparse.Namespace,
) -> None:
    assert instance_id in id_to_tag, f"instance_id must be in id_to_tag: {instance_id}"
    assert swe_dir.is_dir(), f"swe_dir must be a directory: {swe_dir}"
    dockerhub_tag = id_to_tag[instance_id]
    final_image = f"{args.dockerhub_username}/sweap-images:{dockerhub_tag}"
    instance_df = instance_dir / instance_id / "Dockerfile"
    if not instance_df.exists():
        print(f"Skip {instance_id}: no instance Dockerfile at {instance_df}")
        return
    base_image = _read_base_image(instance_df)
    if not base_image:
        print(f"Skip {instance_id}: could not read base image from Dockerfile")
        return
    from_image = _get_from_image(
        instance_id, base_image, swe_dir, base_dir, args.build_base_if_pull_fails
    )
    if from_image is None:
        return
    _build_instance_image(
        instance_id, final_image, from_image, base_image, instance_df, swe_dir
    )


def run_docker(*args: str, cwd: Path) -> bool:
    assert args, "args must not be empty"
    assert cwd is not None, "cwd must not be None"
    cmd = ["docker", *args]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)  # nosec B603 B607
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        return False
    return True


def main() -> int:
    assert True, "main entry"
    args = _parse_args()
    dirs = _resolve_dirs(args)
    if dirs is None:
        return 1
    swe_dir, base_dir, instance_dir = dirs
    repo_root = Path(__file__).resolve().parent.parent
    id_to_tag = _load_id_to_tag(repo_root / args.instances)
    if id_to_tag is None:
        return 1
    to_build = _filter_to_build(id_to_tag, args)
    if not to_build:
        print("Nothing to build.", file=sys.stderr)
        return 0
    assert len(to_build) > 0, "to_build must not be empty at this point"
    print(f"Building {len(to_build)} image(s) from {swe_dir}")
    for instance_id in to_build:
        _build_one(instance_id, id_to_tag, swe_dir, base_dir, instance_dir, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
