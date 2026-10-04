"""Read/write the review dataset, which is stored as one gzipped CSV per app per stage
(data/<stage>/<app_slug>.csv.gz) so that no single file exceeds GitHub's 100 MB limit."""
from pathlib import Path

import pandas as pd

from apps import APPS

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
STAGES = ("raw", "clean", "tagged")


def stage_path(stage, slug):
    return DATA_DIR / stage / f"{slug}.csv.gz"


TEXT_COLUMNS = {"review_id": str, "app_version": str, "content": str}


def read_app(stage, slug, **kwargs):
    kwargs.setdefault("dtype", TEXT_COLUMNS)
    return pd.read_csv(stage_path(stage, slug), **kwargs)


def write_app(df, stage, slug):
    path = stage_path(stage, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.rename(path)


def read_stage(stage, columns=None, **kwargs):
    """Concatenate every app's file for a stage, in the canonical APPS order."""
    frames = [read_app(stage, slug, usecols=columns, **kwargs) for slug in APPS if stage_path(stage, slug).exists()]
    return pd.concat(frames, ignore_index=True)
