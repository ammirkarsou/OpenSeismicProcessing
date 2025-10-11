# Golem User Manual

This guide lists the handful of functions you will interact with day to day when using Golem for seismic-processing pipelines. Everything else in the codebase supports these entry points.

---

## 1. Project & Dataset Management (``golem.catalog``)

| Function | Use Case |
| --- | --- |
| `catalog.ensure_project(name, root_path, description="") -> int` | Create/update a project and get its ID. Run once per dataset location. |
| `catalog.show_projects()` | Print all projects currently registered. |
| `catalog.show_datasets(project_id)` | Display the datasets stored for a project (SEG-Y, Zarr, NPY, etc.). |
| `catalog.get_dataset(project_id, name)` | Fetch dataset metadata (contains `file_path`, `file_type`, and any tags you stored). |
| `catalog.register_dataset(project_id, name, file_path, file_type, metadata=None)` | Manually register an asset (usually not needed if you use the pipeline helpers, but useful for ad‑hoc files). |
| `catalog.reset_project(project_name, confirm=True, remove_outputs=False)` | Clear catalog entries for a project while prototyping. |

> **Tip:** dataset metadata is free-form. Add tags like `{"stack_type": "prestack"}` or `{"role": "output"}` to help the UI filter later.

---

## 2. Building & Running Pipelines

| Function | Use Case |
| --- | --- |
| `build_steps(step_specs)` (from `golem.catalog.steps`) | Convert declarative specs such as `{ "name": "write_data", ... }` into pipeline steps. Use this to define your “boxes” in code or the future UI. |
| `catalog.run_simple_pipeline(project_name, pipeline_steps, output_context_key, output_dataset_name=None, output_filetype="unknown", input_datasets=None)` | Execute the steps, track the run in the catalog, and automatically register the output dataset. |

### Dataset placeholders inside step specs

- `"file_path": "dataset:NAME"` → resolves to the stored path for dataset `NAME`.
- `"zarr_store": "dataset:NAME"` → same idea for Zarr datasets.
- `"segy_input": ["dataset:shot1", "dataset:shot2"]` → in functions that accept lists, you can mix literal paths and dataset references.

The pipeline `context` collects outputs you name via `"output": "data"` (for example). Pass that same key to `run_simple_pipeline` as `output_context_key` so the catalog knows which file to register.

---

## 3. SEG-Y → Zarr Workflow (``golem``)

These functions let you explore SEG-Y headers, convert data to Zarr, and read it lazily.

| Function | Description |
| --- | --- |
| `golem.preview_segy_headers(context, segy_paths, headers=None, n_traces=1000)` | Return a Pandas DataFrame with stats (min/max/mean/std/unique) for every SEG-Y header. Use it to decide which headers to keep. |
| `golem.segy_directory_to_zarr(context, segy_input, zarr_out, headers=None, chunk_trace=512)` | Convert one or many SEG-Y files into a Zarr store. Stores chosen headers as arrays and saves the original binary/text headers in Zarr metadata. |
| `golem.preview_zarr_headers(context, zarr_store, headers=None, n_traces=1000)` | Inspect header behaviour inside an existing Zarr store; results (stats + optional figure) land in the context. |
| `golem.load_zarr_datasets(context, zarr_store, headers=None, include_amplitude=True, output="zarr_data", eager=False)` | Open the Zarr arrays lazily. Pass a catalog dataset (the dict from `get_dataset`) or a direct path. Set `eager=True` only if you want NumPy copies. |
| `golem.extract_zarr_text_headers(context, zarr_store)` | Fetch the SEG-Y text headers that were stored during conversion. |
| `golem.extract_zarr_binary_headers(context, zarr_store)` | Fetch the SEG-Y binary headers stored in metadata. |

---

## 4. Basic I/O Helpers (``golem``)

| Function | Description |
| --- | --- |
| `golem.get_trace_data(context, file_path, ignore_geometry=True)` | Load SEG-Y traces into memory (single file or directory). Useful for quick conversions. |
| `golem.write_data(context, file_folder, file_name, format="npy")` | Save `context["data"]` as `.npy` or binary. Pipeline-friendly; pair it with `run_simple_pipeline` to catalogue the output. |

Most other functions in the repository are lower-level utilities behind these entry points and do not need to be called directly.

---

## 5. Typical Usage Snippet

```python
from golem import catalog
from golem.catalog.steps import build_steps

# 1. Ensure project exists
project_id = catalog.ensure_project("BP 2004", "/home/user/BP2004")

# 2. Preview headers
import golem
df = golem.preview_segy_headers({}, "/home/user/BP2004/ShotGather")
print(df.head())

# 3. Convert to Zarr and register output
definition = build_steps([
    {"name": "segy_directory_to_zarr",
     "segy_input": "/home/user/BP2004/ShotGather",
     "zarr_out": "/home/user/BP2004/zarr/shotgather.zarr",
     "headers": ["SourceX", "GroupX", "offset"],
     "output": "zarr_store"}
])

catalog.run_simple_pipeline(
    project_name="BP 2004",
    pipeline_steps=definition,
    output_context_key="zarr_store",
    output_dataset_name="shotgather_zarr",
    output_filetype="zarr"
)

# 4. Load data lazily
ctx = {}
arrays = golem.load_zarr_datasets(ctx, catalog.get_dataset(project_id, "shotgather_zarr"))
amplitude = arrays["amplitude"]      # still a Zarr array
```

That’s it—these are the handful of functions to remember. Build step specs, run them through `run_simple_pipeline`, and let the catalog keep track of everything you load or produce.
