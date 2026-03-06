"""Build SWE-bench Pro Docker images locally."""

from __future__ import annotations

import json
import re
import subprocess  # nosec B404
import sys
from pathlib import Path


def local_base_tag(base_name: str) -> str:
    assert base_name is not None, "base_name must not be None"
    assert isinstance(base_name, str), "base_name must be a string"
    return "sweap-local/" + base_name.lower()


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
    if run_docker("pull", base_image, cwd=swe_dir):
        return base_image
    if not build_base:
        print(f"Skip {instance_id}: base image pull failed and build_base=False")
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
    dockerhub_username: str,
    build_base_if_pull_fails: bool,
) -> None:
    assert instance_id in id_to_tag, f"instance_id must be in id_to_tag: {instance_id}"
    assert swe_dir.is_dir(), f"swe_dir must be a directory: {swe_dir}"
    dockerhub_tag = id_to_tag[instance_id]
    final_image = f"{dockerhub_username}/sweap-images:{dockerhub_tag}"
    instance_df = instance_dir / instance_id / "Dockerfile"
    if not instance_df.exists():
        print(f"Skip {instance_id}: no instance Dockerfile at {instance_df}")
        return
    base_image = _read_base_image(instance_df)
    if not base_image:
        print(f"Skip {instance_id}: could not read base image from Dockerfile")
        return
    from_image = _get_from_image(
        instance_id, base_image, swe_dir, base_dir, build_base_if_pull_fails
    )
    if from_image is None:
        return
    _build_instance_image(
        instance_id, final_image, from_image, base_image, instance_df, swe_dir
    )


def load_id_to_tag(instances_file: Path) -> dict[str, str]:
    assert instances_file is not None, "instances_file must not be None"
    assert isinstance(instances_file, Path), "instances_file must be a Path"
    if not instances_file.exists():
        raise FileNotFoundError(f"Instances file not found: {instances_file}")
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


def build_images(
    swe_bench_pro_dir: Path,
    instances_file: Path,
    dockerhub_username: str = "jefzda",
    instance_ids: list[str] | None = None,
    limit: int | None = None,
    build_base_if_pull_fails: bool = True,
) -> None:
    """Build sweap Docker images locally and tag as <dockerhub_username>/sweap-images:<tag>."""
    assert isinstance(swe_bench_pro_dir, Path), "swe_bench_pro_dir must be a Path"
    assert isinstance(instances_file, Path), "instances_file must be a Path"
    swe_dir = swe_bench_pro_dir.resolve()
    if not swe_dir.is_dir():
        raise FileNotFoundError(f"SWE-bench Pro dir not found: {swe_dir}")
    base_dir = swe_dir / "dockerfiles" / "base_dockerfile"
    instance_dir = swe_dir / "dockerfiles" / "instance_dockerfile"
    if not base_dir.is_dir() or not instance_dir.is_dir():
        raise FileNotFoundError(f"dockerfiles not found under {swe_dir}")
    id_to_tag = load_id_to_tag(instances_file)
    to_build = list(id_to_tag.keys())
    if instance_ids is not None:
        missing = set(instance_ids) - set(to_build)
        if missing:
            print(
                f"Warning: instances not in instances file: {missing}", file=sys.stderr
            )
        to_build = [i for i in to_build if i in instance_ids]
    if limit is not None:
        to_build = to_build[:limit]
    if not to_build:
        print("Nothing to build.", file=sys.stderr)
        return
    print(f"Building {len(to_build)} image(s) from {swe_dir}")
    for instance_id in to_build:
        _build_one(
            instance_id,
            id_to_tag,
            swe_dir,
            base_dir,
            instance_dir,
            dockerhub_username,
            build_base_if_pull_fails,
        )
