"""
Catalog utilities for tracking seismic processing projects, jobs, and artifacts.

These helpers maintain a lightweight SQLite database that stores metadata about
each project, pipeline run, and generated file while leaving the heavy data on disk.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List

import platformdirs

__all__ = [
    "DB_PATH",
    "init_db",
    "get_workspace_root",
    "set_workspace_root",
    "ensure_project",
    "delete_project",
    "rename_project",
    "get_project",
    "register_dataset",
    "delete_dataset",
    "get_dataset",
    "get_datasets",
    "list_datasets",
    "show_projects",
    "show_datasets",
    "run_project_pipeline",
    "start_job",
    "finish_job",
    "log_artifact",
    "list_projects",
    "list_jobs",
    "list_artifacts",
    "run_simple_pipeline",
    "reset_project",
]

# Cross-platform user-level config + database locations (platformdirs).
_APP_NAME = "OpenSeismicProcessing"
_CONFIG_DIR = Path(platformdirs.user_config_dir(_APP_NAME))
_CONFIG_PATH = _CONFIG_DIR / "config.json"
_DATA_DIR = Path(platformdirs.user_data_dir(_APP_NAME))
_DB_SUBDIR = "catalog"
_DB_NAME = "golem_catalog.db"
_DB_PATH = _DATA_DIR / _DB_SUBDIR / _DB_NAME
# Legacy locations from previous releases.
_LEGACY_CONFIG_DIR = Path.home() / ".openseismicprocessing"
_LEGACY_CONFIG_PATH = _LEGACY_CONFIG_DIR / "config.json"
_LEGACY_DB_PATHS = [
    _LEGACY_CONFIG_DIR / _DB_SUBDIR / _DB_NAME,  # ~/.openseismicprocessing/catalog/golem_catalog.db
    Path(__file__).resolve().parents[3] / _DB_SUBDIR / _DB_NAME,  # repo-root/catalog/golem_catalog.db
]


def _load_config() -> dict[str, Any]:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except Exception:
            return {}
    # Fallback to legacy config path if present.
    if _LEGACY_CONFIG_PATH.exists():
        try:
            return json.loads(_LEGACY_CONFIG_PATH.read_text())
        except Exception:
            return {}
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return {}


def _save_config(config: dict[str, Any]) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(config, indent=2))


def _compute_db_path(workspace_root: Optional[str]) -> Path:
    # Workspace root is stored for the UI, but DB always lives in user data dir.
    return _DB_PATH


def _maybe_migrate_legacy_db(target: Path) -> None:
    """Copy an existing legacy DB (repo-root) into the new location if missing."""
    if target.exists():
        return
    candidates: list[Path] = []
    # Legacy workspace-root location (if old config stored it).
    try:
        legacy_cfg = {}
        if _LEGACY_CONFIG_PATH.exists():
            legacy_cfg = json.loads(_LEGACY_CONFIG_PATH.read_text())
        root = legacy_cfg.get("workspace_root")
        if root:
            candidates.append(Path(root).expanduser() / _DB_SUBDIR / _DB_NAME)
    except Exception:
        pass
    candidates.extend(_LEGACY_DB_PATHS)
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.resolve() != target.resolve():
                shutil.copy2(candidate, target)
                return
        except OSError:
            # If migration fails, we'll create a fresh DB on next init.
            pass


def _read_workspace_root_from_config() -> Optional[str]:
    cfg = _load_config()
    root = cfg.get("workspace_root")
    if root:
        return str(Path(root).expanduser())
    return None


def _get_db_path() -> Path:
    path = _compute_db_path(_read_workspace_root_from_config())
    path.parent.mkdir(parents=True, exist_ok=True)
    _maybe_migrate_legacy_db(path)
    global DB_PATH
    DB_PATH = path
    return path


# Exported for compatibility; updated when workspace root changes.
DB_PATH = _compute_db_path(_read_workspace_root_from_config())


def _connect() -> sqlite3.Connection:
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    """Add a column to an existing table if it is missing (simple schema migration)."""
    existing = {
        row[1]  # pragma table_info: 0=id, 1=name, 2=type, ...
        for row in conn.execute(f"PRAGMA table_info({table})")
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _resolve_dataset_token(value: Any, datasets: dict[str, dict[str, Any]]) -> Any:
    """Resolve dataset placeholders inside pipeline step kwargs."""
    if isinstance(value, str):
        if value.startswith("dataset:"):
            dataset_name = value.split(":", 1)[1]
            dataset = datasets.get(dataset_name)
            if dataset is None:
                raise ValueError(f"Dataset '{dataset_name}' not found in project catalog.")
            return dataset["file_path"]
        if value.startswith("dataset_meta:"):
            dataset_name = value.split(":", 1)[1]
            dataset = datasets.get(dataset_name)
            if dataset is None:
                raise ValueError(f"Dataset '{dataset_name}' not found in project catalog.")
            return dataset
        return value

    if isinstance(value, list):
        return [_resolve_dataset_token(item, datasets) for item in value]

    if isinstance(value, tuple):
        return tuple(_resolve_dataset_token(item, datasets) for item in value)

    if isinstance(value, dict):
        return {k: _resolve_dataset_token(v, datasets) for k, v in value.items()}

    return value


def init_db() -> None:
    """Create tables if they are missing."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                root_path TEXT NOT NULL,
                description TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                description TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(project_id, file_path)
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                job_type TEXT NOT NULL,
                params_json TEXT,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                UNIQUE(project_id, name)
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                step_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        # Ensure newer columns exist when working with catalogs created before metadata support.
        _ensure_column(conn, "projects", "metadata_json", "TEXT")
        _ensure_column(conn, "datasets", "metadata_json", "TEXT")
        _ensure_column(conn, "artifacts", "metadata_json", "TEXT")
        # metadata table may exist without updated_at (older catalogs)
        _ensure_column(conn, "metadata", "updated_at", "TEXT")


def _utcnow() -> str:
    return datetime.utcnow().isoformat()


def get_workspace_root(default: Optional[str] = None) -> Optional[str]:
    """Return the workspace root folder stored in the local config."""
    root = _read_workspace_root_from_config()
    return root if root else default


def set_workspace_root(path: Path | str) -> None:
    """Persist the workspace root folder and ensure a catalog database exists there."""
    value = str(Path(path).expanduser().resolve())
    cfg = _load_config()
    cfg["workspace_root"] = value
    _save_config(cfg)
    global DB_PATH
    DB_PATH = _compute_db_path(value)
    # Create tables in the new location immediately.
    init_db()


def ensure_project(
    name: str,
    root_path: Path,
    description: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> int:
    """
    Fetch an existing project ID or create one if missing.
    """
    init_db()
    ts = _utcnow()
    with _connect() as conn:
        cur = conn.execute("SELECT id FROM projects WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            project_id = row[0]
            conn.execute(
                """
                UPDATE projects
                SET root_path = ?, description = ?, metadata_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(root_path), description, json.dumps(metadata or {}), ts, project_id),
            )
            return project_id

        cur = conn.execute(
            """
            INSERT INTO projects (name, root_path, description, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, str(root_path), description, json.dumps(metadata or {}), ts, ts),
        )
        return cur.lastrowid


def delete_project(*, name: Optional[str] = None, project_id: Optional[int] = None) -> None:
    """Remove a project (survey) from the catalog."""
    if name is None and project_id is None:
        raise ValueError("Either 'name' or 'project_id' must be provided.")

    init_db()
    with _connect() as conn:
        if project_id is not None:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        else:
            conn.execute("DELETE FROM projects WHERE name = ?", (name,))

def rename_project(old_name: str, new_name: str, new_root: Path) -> None:
    """
    Rename a project and update its root path while preserving datasets/jobs/artifacts.
    """
    if not old_name or not new_name:
        raise ValueError("Both old_name and new_name are required.")
    init_db()
    ts = _utcnow()
    with _connect() as conn:
        # ensure target name not taken
        row = conn.execute("SELECT id FROM projects WHERE name = ?", (new_name,)).fetchone()
        if row:
            raise ValueError(f"Project '{new_name}' already exists.")
        res = conn.execute(
            "UPDATE projects SET name = ?, root_path = ?, updated_at = ? WHERE name = ?",
            (new_name, str(new_root), ts, old_name),
        )
        if res.rowcount == 0:
            raise ValueError(f"Project '{old_name}' not found.")


def get_project(*, name: Optional[str] = None, project_id: Optional[int] = None) -> dict[str, Any]:
    """
    Fetch a project by name or ID. Raises ValueError if not found.
    """
    if name is None and project_id is None:
        raise ValueError("Either 'name' or 'project_id' must be provided.")

    init_db()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        if project_id is not None:
            row = conn.execute(
                "SELECT id, name, root_path, description, metadata_json, created_at, updated_at FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, name, root_path, description, metadata_json, created_at, updated_at FROM projects WHERE name = ?",
                (name,),
            ).fetchone()

    if row is None:
        key = f"id={project_id}" if project_id is not None else f"name='{name}'"
        raise ValueError(f"Project with {key} not found in catalog.")

    data = dict(row)
    metadata = data.get("metadata_json")
    data["metadata"] = json.loads(metadata) if metadata else {}
    del data["metadata_json"]
    return data


def register_dataset(
    project_id: int,
    name: str,
    file_path: Path,
    file_type: str,
    description: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> int:
    """
    Register or update a dataset for a project and return its ID.

    The optional ``metadata`` dictionary can store attributes such as stack type
    (e.g. ``{"stack_type": "prestack"}``) or acquisition notes. The data is
    persisted as JSON in the catalog.
    """
    init_db()
    ts = _utcnow()
    with _connect() as conn:
        cur = conn.execute(
            "SELECT id FROM datasets WHERE project_id = ? AND file_path = ?",
            (project_id, str(file_path)),
        )
        row = cur.fetchone()
        if row:
            dataset_id = row[0]
            conn.execute(
                """
                UPDATE datasets
                SET name = ?, file_type = ?, description = ?, metadata_json = ?, created_at = ?
                WHERE id = ?
                """,
                (name, file_type, description, json.dumps(metadata or {}), ts, dataset_id),
            )
            return dataset_id

        cur = conn.execute(
            """
            INSERT INTO datasets (project_id, name, file_path, file_type, description, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (project_id, name, str(file_path), file_type, description, json.dumps(metadata or {}), ts),
        )
        return cur.lastrowid


def delete_dataset(
    project_id: int,
    name: str,
    *,
    remove_file: bool = False,
    confirm: bool = False,
) -> dict[str, Any]:
    """Remove a dataset entry (and optionally its file) from the catalog.

    Parameters
    ----------
    project_id : int
        Project identifier.
    name : str
        Dataset name to remove.
    remove_file : bool, default False
        When True, delete the file from disk if it exists.
    confirm : bool, default False
        Must be True to proceed; acts as an explicit safety check.

    Returns
    -------
    dict
        Metadata of the removed dataset (useful for logging).
    """

    if not confirm:
        raise ValueError("Set confirm=True to delete a dataset entry.")

    dataset = get_dataset(project_id, name)

    with _connect() as conn:
        conn.execute("DELETE FROM datasets WHERE project_id = ? AND name = ?", (project_id, name))
        conn.execute(
            "DELETE FROM artifacts WHERE project_id = ? AND file_path = ?",
            (project_id, dataset["file_path"]),
        )

    if remove_file:
        path = Path(dataset["file_path"])
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            raise RuntimeError(f"Failed to remove file {path}: {exc}") from exc

    return dataset


def get_dataset(project_id: int, name: str) -> dict[str, Any]:
    """
    Fetch a dataset (input) by name for a given project. Raises ValueError if not found.
    """
    init_db()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, name, file_path, file_type, description, metadata_json, created_at
            FROM datasets
            WHERE project_id = ? AND name = ?
            """,
            (project_id, name),
        ).fetchone()

    if row is None:
        raise ValueError(f"Dataset '{name}' not registered for project id={project_id}.")

    data = dict(row)
    metadata = data.get("metadata_json")
    data["metadata"] = json.loads(metadata) if metadata else {}
    del data["metadata_json"]
    return data


def start_job(
    project_id: int,
    name: str,
    job_type: str,
    params: Optional[dict[str, Any]] = None,
) -> int:
    """
    Register a new job (pipeline run) for a project and return its ID.
    """
    init_db()
    ts = _utcnow()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO jobs (project_id, name, job_type, params_json, status, started_at, finished_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL)
            """,
            (project_id, name, job_type, json.dumps(params or {}), "running", ts),
        )
        return cur.lastrowid


def finish_job(job_id: int, status: str = "completed") -> None:
    """
    Mark a job as completed (or failed) and stamp the finish time.
    """
    ts = _utcnow()
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, finished_at = ? WHERE id = ?",
            (status, ts, job_id),
        )


def log_artifact(
    project_id: int,
    job_id: int,
    step_name: str,
    file_path: Path,
    file_type: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """
    Record an artifact produced by a job step.
    """
    ts = _utcnow()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO artifacts (project_id, job_id, step_name, file_path, file_type, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                job_id,
                step_name,
                str(file_path),
                file_type,
                json.dumps(metadata or {}),
                ts,
            ),
        )


def list_projects() -> list[dict[str, Any]]:
    """Return all registered projects."""
    init_db()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, name, root_path, description, metadata_json, created_at, updated_at FROM projects ORDER BY created_at"
        ).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            metadata = data.get("metadata_json")
            data["metadata"] = json.loads(metadata) if metadata else {}
            del data["metadata_json"]
            result.append(data)
        return result


def list_jobs(project_id: int) -> list[dict[str, Any]]:
    """Return jobs for a given project."""
    init_db()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, name, job_type, status, started_at, finished_at, params_json
            FROM jobs
            WHERE project_id = ?
            ORDER BY started_at DESC
            """,
            (project_id,),
        ).fetchall()
        jobs = []
        for row in rows:
            data = dict(row)
            params = data.get("params_json")
            data["params"] = json.loads(params) if params else {}
            del data["params_json"]
            jobs.append(data)
        return jobs


def list_datasets(project_id: int) -> list[dict[str, Any]]:
    """Return datasets registered under a project."""
    init_db()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, name, file_path, file_type, description, metadata_json, created_at
            FROM datasets
            WHERE project_id = ?
            ORDER BY name
            """,
            (project_id,),
        ).fetchall()
        datasets: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            metadata = data.get("metadata_json")
            data["metadata"] = json.loads(metadata) if metadata else {}
            del data["metadata_json"]
            datasets.append(data)
        return datasets


def get_datasets(
    project_id: int,
    names: Optional[list[str]] = None,
    file_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Fetch multiple datasets. Optionally filter by names or file type.
    """
    datasets = list_datasets(project_id)
    if names is not None:
        selected = {name: False for name in names}
        filtered = []
        for ds in datasets:
            if ds["name"] in selected:
                filtered.append(ds)
                selected[ds["name"]] = True
        missing = [name for name, found in selected.items() if not found]
        if missing:
            raise ValueError(f"Datasets not found for project id={project_id}: {missing}")
        datasets = filtered

    if file_type is not None:
        datasets = [ds for ds in datasets if ds["file_type"] == file_type]

    return datasets


def list_artifacts(job_id: int) -> list[dict[str, Any]]:
    """Return artifacts generated by a job."""
    init_db()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, step_name, file_path, file_type, metadata_json, created_at
            FROM artifacts
            WHERE job_id = ?
            ORDER BY created_at
            """,
            (job_id,),
        ).fetchall()
        artifacts = []
        for row in rows:
            data = dict(row)
            metadata = data.get("metadata_json")
            data["metadata"] = json.loads(metadata) if metadata else {}
            del data["metadata_json"]
            artifacts.append(data)
        return artifacts


def show_projects() -> None:
    """Print all catalogued projects."""
    projects = list_projects()
    if not projects:
        print("No projects registered.")
        return
    for project in projects:
        print(f"[{project['id']}] {project['name']} -> {project['root_path']}")


def show_datasets(project_id: int) -> None:
    """Print datasets registered for a project."""
    datasets = list_datasets(project_id)
    if not datasets:
        print(f"No datasets registered for project id={project_id}.")
        return
    for dataset in datasets:
        metadata = dataset.get("metadata") or {}
        meta_str = f" | metadata: {metadata}" if metadata else ""
        print(
            f" - {dataset['name']} ({dataset['file_type']}): {dataset['file_path']}{meta_str}"
        )


def run_project_pipeline(
    project_name: str,
    steps: List[tuple],
    *,
    job_name: Optional[str] = None,
    job_type: str = "pipeline",
    input_datasets: Optional[List[str]] = None,
    artifact_specs: Optional[List[dict[str, Any]]] = None,
    auto_log_inputs: bool = True,
) -> dict[str, Any]:
    """Execute a pipeline for a catalogued project and record the run.

    Parameters
    ----------
    project_name : str
        Name of the project registered in the catalog.
    steps : list
        Pipeline steps compatible with :func:`golem.pipeline.run_pipeline`. Any
        kwarg value starting with ``dataset:NAME`` is automatically resolved to
        the file path of the registered dataset ``NAME``. Use ``dataset_meta:NAME``
        to inject the complete dataset record instead.
    job_name : str, optional
        Identifier stored in the catalog ``jobs`` table. Defaults to a timestamped
        string based on ``job_type`` when omitted.
    job_type : str, default "pipeline"
        Type label for the job entry (e.g. "segy_to_npy").
    input_datasets : list[str], optional
        Datasets to record as inputs for this job. Their metadata is merged with
        the logged artifact entry.
    artifact_specs : list[dict], optional
        Definitions describing which outputs to log as artifacts. Each dict should
        include ``context_key`` (key in the pipeline context containing a path)
        and ``file_type``. Additional optional keys:

        - ``metadata``: dict merged into the artifact metadata.
        - ``source_dataset``: dataset name whose metadata should be merged.
        - ``dataset_name``: register the emitted file as a new dataset under this name.
        - ``register_dataset``: bool flag (default False) that enables auto-registration.
        - ``description``: optional dataset description when auto-registering.
    auto_log_inputs : bool, default True
        When True, datasets referenced via ``input_datasets`` are automatically
        logged as artifacts with ``role='input'`` metadata.

    Returns
    -------
    dict
        The pipeline context returned by :func:`golem.pipeline.run_pipeline`.
    """

    from golem.pipeline import run_pipeline  # local import to avoid circular deps

    project = get_project(name=project_name)
    project_id = project["id"]

    dataset_entries = get_datasets(project_id)
    dataset_map = {entry["name"]: entry for entry in dataset_entries}

    resolved_steps: List[tuple] = []
    for func, kwargs in steps:
        resolved_kwargs = {
            key: _resolve_dataset_token(value, dataset_map)
            for key, value in kwargs.items()
        }
        resolved_steps.append((func, resolved_kwargs))

    if job_name is None:
        job_name = f"{job_type}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"

    job_id = start_job(
        project_id=project_id,
        name=job_name,
        job_type=job_type,
        params={"project": project_name},
    )

    status = "completed"
    try:
        context = run_pipeline(resolved_steps)

        errors = [value for key, value in context.items() if key.endswith("_error")]
        if errors:
            status = f"failed: {errors[-1]}"
            return context

        # Log input datasets if requested.
        if auto_log_inputs and input_datasets:
            for dataset_name in input_datasets:
                dataset = dataset_map.get(dataset_name)
                if dataset is None:
                    raise ValueError(
                        f"Dataset '{dataset_name}' not registered for project '{project_name}'."
                    )
                meta = {"role": "input", **(dataset.get("metadata") or {})}
                log_artifact(
                    project_id=project_id,
                    job_id=job_id,
                    step_name=f"input::{dataset_name}",
                    file_path=Path(dataset["file_path"]),
                    file_type=dataset["file_type"],
                    metadata=meta,
                )

        # Log outputs specified by artifact_specs.
        for spec in artifact_specs or []:
            context_key = spec.get("context_key")
            if not context_key:
                continue
            path_value = context.get(context_key)
            if not path_value:
                continue

            metadata = dict(spec.get("metadata") or {})
            source_dataset = spec.get("source_dataset")
            if source_dataset:
                dataset = dataset_map.get(source_dataset)
                if dataset:
                    metadata = {**(dataset.get("metadata") or {}), **metadata}

            artifact_path = Path(path_value)
            log_artifact(
                project_id=project_id,
                job_id=job_id,
                step_name=spec.get("step_name", context_key),
                file_path=artifact_path,
                file_type=spec.get("file_type", "unknown"),
                metadata=metadata,
            )

            if spec.get("register_dataset"):
                dataset_name = spec.get("dataset_name") or spec.get("step_name") or context_key
                description = spec.get("description", "")
                dataset_metadata = dict(metadata)
                try:
                    register_dataset(
                        project_id=project_id,
                        name=dataset_name,
                        file_path=artifact_path,
                        file_type=spec.get("file_type", "unknown"),
                        description=description,
                        metadata=dataset_metadata,
                    )
                    dataset_map[dataset_name] = {
                        "name": dataset_name,
                        "file_path": str(artifact_path),
                        "file_type": spec.get("file_type", "unknown"),
                        "metadata": dataset_metadata,
                    }
                except Exception as exc:
                    print(f"⚠️ Failed to register dataset '{dataset_name}': {exc}")

        return context

    except Exception as exc:
        status = f"failed: {exc}"
        raise
    finally:
        finish_job(job_id, status=status)


def run_simple_pipeline(
    project_name: str,
    *,
    pipeline_steps: list[tuple],
    output_context_key: str,
    output_dataset_name: Optional[str] = None,
    output_filetype: str = "unknown",
    output_description: str = "",
    input_datasets: Optional[list[str]] = None,
    job_type: str = "pipeline",
    job_name: Optional[str] = None,
) -> dict[str, Any]:
    """Convenience wrapper around :func:`run_project_pipeline`.

    Parameters
    ----------
    project_name : str
        Name of the project registered in the catalog.
    pipeline_steps : list
        List of ``(callable, kwargs)`` pairs ready for :func:`run_project_pipeline`.
    output_context_key : str
        The context key produced by the pipeline (e.g. the output of ``write_data``)
        whose path should be logged and registered.
    output_dataset_name : str, optional
        Dataset name under which the resulting file should be registered. Defaults
        to ``output_context_key``.
    output_filetype : str, default \"unknown\"
        File type label (e.g. ``\"npy\"``).
    output_description : str, optional
        Human-readable description stored alongside the dataset metadata.
    input_datasets : list[str], optional
        Input dataset names to log. Defaults to none.
    job_type : str, default "pipeline"
        Job type label stored in the catalog.
    job_name : str, optional
        Explicit job name; otherwise a timestamped string is generated.

    Returns
    -------
    dict
        Pipeline context produced by :func:`run_project_pipeline`.
    """

    dataset_name = output_dataset_name or output_context_key

    artifact_specs = [
        {
            "context_key": output_context_key,
            "file_type": output_filetype,
            "dataset_name": dataset_name,
            "register_dataset": True,
            "description": output_description,
        }
    ]

    return run_project_pipeline(
        project_name=project_name,
        steps=pipeline_steps,
        job_type=job_type,
        job_name=job_name,
        input_datasets=input_datasets,
        artifact_specs=artifact_specs,
    )


def reset_project(project_name: str, *, confirm: bool = False, remove_outputs: bool = False) -> None:
    """Delete all catalog metadata (and optionally files) for a project."""

    if not confirm:
        raise ValueError("Set confirm=True to reset a project.")

    project = get_project(name=project_name)
    project_id = project["id"]

    dataset_entries = list_datasets(project_id)
    artifact_entries = []
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        artifact_entries = conn.execute(
            "SELECT file_path FROM artifacts WHERE project_id = ?",
            (project_id,),
        ).fetchall()

    with _connect() as conn:
        conn.execute("DELETE FROM artifacts WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM datasets WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM jobs WHERE project_id = ?", (project_id,))

    if remove_outputs:
        for entry in artifact_entries:
            path = Path(entry[0])
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
        for entry in dataset_entries:
            path = Path(entry["file_path"])
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
