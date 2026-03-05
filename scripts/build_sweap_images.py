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


def main() -> int:
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
        "--limit",
        type=int,
        default=None,
        help="Build only the first N instances",
    )
    parser.add_argument(
        "--instance-ids",
        type=str,
        nargs="*",
        default=None,
        help="Build only these instance_ids",
    )
    parser.add_argument(
        "--dockerhub-username",
        type=str,
        default="jefzda",
        help="Tag images as <username>/sweap-images:<tag>",
    )
    parser.add_argument(
        "--build-base-if-pull-fails",
        action="store_true",
        default=True,
        help="If base image pull fails, build base from base_dockerfile (default: True)",
    )
    parser.add_argument(
        "--no-build-base-if-pull-fails",
        action="store_false",
        dest="build_base_if_pull_fails",
        help="Do not build base from base_dockerfile when pull fails",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    swe_dir = (repo_root / args.swe_bench_pro_dir).resolve()
    instances_file = repo_root / args.instances

    if not swe_dir.is_dir():
        print(f"Error: SWE-bench Pro dir not found: {swe_dir}", file=sys.stderr)
        return 1
    dockerfiles_dir = swe_dir / "dockerfiles"
    base_dir = dockerfiles_dir / "base_dockerfile"
    instance_dir = dockerfiles_dir / "instance_dockerfile"
    if not base_dir.is_dir() or not instance_dir.is_dir():
        print(f"Error: dockerfiles not found under {swe_dir}", file=sys.stderr)
        return 1

    # Load instance_id -> dockerhub_tag from instances.jsonl (unique by instance_id)
    id_to_tag: dict[str, str] = {}
    if not instances_file.exists():
        print(f"Error: instances file not found: {instances_file}", file=sys.stderr)
        return 1
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

    to_build = list(id_to_tag.keys())
    if args.instance_ids is not None:
        to_build = [i for i in to_build if i in args.instance_ids]
        if len(to_build) != len(args.instance_ids):
            missing = set(args.instance_ids) - set(to_build)
            print(f"Warning: instances not in {instances_file}: {missing}", file=sys.stderr)
    if args.limit is not None:
        to_build = to_build[: args.limit]

    if not to_build:
        print("Nothing to build.", file=sys.stderr)
        return 0

    print(f"Building {len(to_build)} image(s) from {swe_dir}")

    # Local base image tag: use a namespace so Docker accepts it (no library/ implied)
    LOCAL_BASE_PREFIX = "sweap-local/"

    def local_base_tag(base_name: str) -> str:
        """Produce a valid local image tag (lowercase, with slash)."""
        return LOCAL_BASE_PREFIX + base_name.lower()

    for instance_id in to_build:
        dockerhub_tag = id_to_tag[instance_id]
        final_image = f"{args.dockerhub_username}/sweap-images:{dockerhub_tag}"

        instance_dockerfile_path = instance_dir / instance_id / "Dockerfile"
        if not instance_dockerfile_path.exists():
            print(f"Skip {instance_id}: no instance Dockerfile at {instance_dockerfile_path}")
            continue

        # Get base image name from instance Dockerfile
        base_image = None
        with instance_dockerfile_path.open() as f:
            for line in f:
                if line.strip().startswith("FROM "):
                    base_image = line.strip().split(maxsplit=1)[1].strip()
                    break
        if not base_image:
            print(f"Skip {instance_id}: could not read base image from Dockerfile")
            continue

        # Ensure base image exists: try pull, else build from base_dockerfile
        base_dockerfile_path = base_dir / instance_id / "Dockerfile"
        pull_ok = run_docker("pull", base_image, cwd=swe_dir)
        from_image = base_image  # FROM to use in instance build
        if not pull_ok and args.build_base_if_pull_fails and base_dockerfile_path.exists():
            print(f"Building base image for {instance_id}...")
            content = base_dockerfile_path.read_text()
            # Replace ECR base with Docker Hub equivalent for local build
            new_content = re.sub(
                r"084828598639\.dkr\.ecr\.us-west-2\.amazonaws\.com/docker-hub/library/",
                "docker.io/library/",
                content,
            )
            if new_content == content:
                new_content = re.sub(
                    r"084828598639\.dkr\.ecr\.us-west-2\.amazonaws\.com/",
                    "",
                    content,
                    count=1,
                )
            local_base = local_base_tag(base_image)
            tmp_base = swe_dir / "dockerfiles" / ".build_base.Dockerfile"
            tmp_base.parent.mkdir(parents=True, exist_ok=True)
            tmp_base.write_text(new_content)
            try:
                if not run_docker(
                    "build",
                    "-f",
                    str(tmp_base.relative_to(swe_dir)),
                    "-t",
                    local_base,
                    ".",
                    cwd=swe_dir,
                ):
                    print(f"Failed to build base for {instance_id}")
                    continue
            finally:
                tmp_base.unlink(missing_ok=True)
            print(f"Built base and tagged as {local_base}")
            from_image = local_base
        elif not pull_ok:
            print(f"Skip {instance_id}: base image pull failed and --no-build-base-if-pull-fails")
            continue

        # Build instance image (use temp Dockerfile if we built local base so FROM is valid)
        instance_df_path = instance_dockerfile_path
        tmp_instance_df = None
        if from_image != base_image:
            instance_content = instance_dockerfile_path.read_text()
            new_from = re.sub(
                r"^FROM\s+\S+",
                f"FROM {from_image}",
                instance_content,
                count=1,
            )
            tmp_instance_df = swe_dir / "dockerfiles" / ".build_instance.Dockerfile"
            tmp_instance_df.write_text(new_from)
            instance_df_path = tmp_instance_df
        try:
            print(f"Building {final_image} ...")
            if not run_docker(
                "build",
                "-f",
                str(instance_df_path.relative_to(swe_dir)),
                "-t",
                final_image,
                ".",
                cwd=swe_dir,
            ):
                print(f"Failed to build {final_image}")
                continue
            print(f"Built and tagged {final_image}")
        finally:
            if tmp_instance_df is not None and tmp_instance_df.exists():
                tmp_instance_df.unlink(missing_ok=True)

    return 0


def run_docker(*args: str, cwd: Path) -> bool:
    cmd = ["docker", *args]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)  # nosec B603 B607
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.stdout and result.returncode != 0:
            print(result.stdout, file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
