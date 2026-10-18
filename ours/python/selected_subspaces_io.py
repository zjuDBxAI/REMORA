"""Read the selected-subspace metadata used by coverage baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectedSubspacesArtifact:
    artifact_id: int
    node_count: int
    subspaces: int
    dsub: int
    refined_subspaces: int
    selected_subspaces: np.ndarray
    subspace_risks: np.ndarray


def read_selected_subspaces(path: str) -> SelectedSubspacesArtifact:
    """Read selection metadata from the historical selection NPZ artifact."""
    with np.load(path, allow_pickle=False) as state:
        artifact = SelectedSubspacesArtifact(
            artifact_id=int(state["artifact_id"]),
            node_count=int(state["node_count"]),
            subspaces=int(state["subspaces"]),
            dsub=int(state["dsub"]),
            refined_subspaces=int(state["refined_subspaces"]),
            selected_subspaces=np.asarray(
                state["selected_subspaces"], dtype=np.uint32
            ),
            subspace_risks=np.asarray(state["subspace_risks"], dtype=np.float64),
        )
    selected = artifact.selected_subspaces.astype(np.int64)
    if (
        artifact.artifact_id <= 0
        or artifact.node_count <= 0
        or artifact.subspaces <= 0
        or artifact.dsub <= 0
        or artifact.refined_subspaces != selected.size
        or artifact.subspace_risks.shape != (artifact.subspaces,)
        or np.any(selected < 0)
        or np.any(selected >= artifact.subspaces)
        or np.unique(selected).size != selected.size
    ):
        raise ValueError("invalid selected-subspace artifact")
    return artifact
