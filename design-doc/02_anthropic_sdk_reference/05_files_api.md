# Files API — Uploading Data to Anthropic Containers

## 1. Overview

The Files API is a beta feature that lets you upload files to Anthropic's servers and reference them across multiple API requests using a persistent `file_id`. This solves a fundamental problem in data analysis: sending large datasets efficiently.

**Without Files API:**
```
Request 1: [full CSV as base64, 500KB]  → Response
Request 2: [full CSV as base64, 500KB]  → Response
Request 3: [full CSV as base64, 500KB]  → Response
Total: 1.5MB of repeated data transfer
```

**With Files API:**
```
Upload: CSV → file_id = "file_01ABC..."
Request 1: [file_id reference, ~50 bytes]  → Response
Request 2: [file_id reference, ~50 bytes]  → Response
Request 3: [file_id reference, ~50 bytes]  → Response
Total: 1 upload + 3 tiny references
```

**Beta Status**: The Files API is in beta as of May 2026. Always include the beta header.

**Required beta header:**
```python
betas=["files-api-2025-04-14"]
```

This header must be included in every request that uses Files API features (upload, reference, or download).

---

## 2. Uploading Files

### Basic Upload Pattern

```python
import anthropic

client = anthropic.Anthropic()

# Upload a CSV file
with open("sales_data.csv", "rb") as f:
    response = client.beta.files.upload(
        file=("sales_data.csv", f, "text/csv"),
    )

file_id = response.id
print(f"Uploaded: {file_id}")           # "file_01XYZ..."
print(f"Filename: {response.filename}") # "sales_data.csv"
print(f"Created: {response.created_at}")
```

### Uploading from Bytes in Memory

```python
import anthropic
import io
import pandas as pd

client = anthropic.Anthropic()

# Generate data in memory and upload
df = pd.DataFrame({
    "month": range(1, 13),
    "revenue": [45000, 52000, 48000, 61000, 58000, 72000,
                69000, 75000, 71000, 83000, 88000, 95000]
})

# Convert to CSV bytes without writing to disk
csv_bytes = df.to_csv(index=False).encode("utf-8")

uploaded = client.beta.files.upload(
    file=("revenue_2025.csv", io.BytesIO(csv_bytes), "text/csv")
)
print(f"Uploaded in-memory data: {uploaded.id}")
```

### Uploading Images

```python
# PNG image
with open("sales_chart.png", "rb") as f:
    chart_file = client.beta.files.upload(
        file=("sales_chart.png", f, "image/png")
    )

# JPEG image
with open("dashboard_screenshot.jpg", "rb") as f:
    screenshot = client.beta.files.upload(
        file=("dashboard.jpg", f, "image/jpeg")
    )
```

### Uploading Text/Markdown/JSON Files

```python
# Plain text
with open("analysis_notes.txt", "rb") as f:
    notes = client.beta.files.upload(
        file=("notes.txt", f, "text/plain")
    )

# JSON data
with open("config.json", "rb") as f:
    config = client.beta.files.upload(
        file=("config.json", f, "application/json")
    )

# Python source file
with open("preprocessing.py", "rb") as f:
    script = client.beta.files.upload(
        file=("preprocessing.py", f, "text/x-python")
    )
```

### MIME Types Reference

| File type | MIME type |
|-----------|-----------|
| CSV | `text/csv` |
| Plain text | `text/plain` |
| JSON | `application/json` |
| PNG | `image/png` |
| JPEG | `image/jpeg` |
| GIF | `image/gif` |
| WebP | `image/webp` |
| PDF | `application/pdf` |
| Python | `text/x-python` |
| Excel | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` |

---

## 3. Using Files in Messages

### Document Content Type

Reference an uploaded file as a document in any message:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Summarise the key trends in this sales data:"
            },
            {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": file_id      # The ID returned by files.upload()
                }
            }
        ]
    }],
    betas=["files-api-2025-04-14"]
)
```

### Multiple Files in One Request

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8192,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare the Q1 and Q2 sales data:"},
            {
                "type": "document",
                "source": {"type": "file", "file_id": q1_file_id}
            },
            {
                "type": "document",
                "source": {"type": "file", "file_id": q2_file_id}
            }
        ]
    }],
    betas=["files-api-2025-04-14"]
)
```

### Image Files

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What anomalies do you see in this chart?"},
            {
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": chart_file_id
                }
            }
        ]
    }],
    betas=["files-api-2025-04-14"]
)
```

### Using Files with Tool Use

Files and tools can be combined in the same request:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8192,
    tools=[{"type": "code_execution_20260120", "name": "code_execution"}],
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyse this dataset and generate a report:"},
            {
                "type": "container_upload",
                "file_id": file_id,
                "filename": "sales_data.csv"
            }
        ]
    }],
    betas=["files-api-2025-04-14"]
)
```

---

## 4. Container Upload

`container_upload` is the content type specifically for making files available inside the code execution sandbox. This is different from `document` (which gives Claude text access) — `container_upload` puts the file in the container's `/files/` directory.

### Basic Container Upload

```python
{
    "type": "container_upload",
    "file_id": "file_01ABC...",
    "filename": "sales_data.csv"   # Path inside container: /files/sales_data.csv
}
```

### Multiple Files for Code Execution

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8192,
    tools=[{"type": "code_execution_20260120", "name": "code_execution"}],
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Join these two datasets and compute summary statistics:"},
            {
                "type": "container_upload",
                "file_id": transactions_file_id,
                "filename": "transactions.csv"    # → /files/transactions.csv
            },
            {
                "type": "container_upload",
                "file_id": products_file_id,
                "filename": "products.csv"         # → /files/products.csv
            }
        ]
    }],
    betas=["files-api-2025-04-14"]
)

# Claude's generated code can then do:
# import pandas as pd
# transactions = pd.read_csv('/files/transactions.csv')
# products = pd.read_csv('/files/products.csv')
# merged = transactions.merge(products, on='product_id')
```

### Document vs Container Upload — When to Use Each

| Need | Content type | Access mode |
|------|-------------|-------------|
| Claude reads CSV content in text | `document` | Text parsing |
| Claude executes code on CSV data | `container_upload` | `/files/filename` |
| Image for visual analysis | `image` | Visual processing |
| Reference material / instructions | `document` | Text context |
| Data for pandas/numpy operations | `container_upload` | File system |

---

## 5. Downloading Generated Files

After code execution, Claude may generate output files (charts, processed CSVs, reports). These can be downloaded via the Files API.

### How Generated Files Work

When Claude's code saves a file inside the container, it becomes available as a downloadable file:

```python
# Claude might execute code like this:
"""
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/files/sales_data.csv')
df['revenue'].plot(kind='bar', title='Monthly Revenue')
plt.tight_layout()
plt.savefig('/files/revenue_chart.png', dpi=150)
print("Chart saved.")

# Save processed data
df_clean = df.dropna()
df_clean.to_csv('/files/clean_data.csv', index=False)
print("Clean data saved.")
"""
```

### Extracting File References from Response

Generated files come back as references in tool result blocks:

```python
def extract_generated_files(response) -> list[dict]:
    """Find file references in code execution output."""
    generated_files = []

    for block in response.content:
        # Tool results from server tools contain the execution output
        if hasattr(block, "content") and isinstance(block.content, list):
            for item in block.content:
                if isinstance(item, dict) and item.get("type") == "file_reference":
                    generated_files.append({
                        "file_id": item.get("file_id"),
                        "filename": item.get("filename"),
                    })

    return generated_files


# Download a generated file
def download_file(client: anthropic.Anthropic, file_id: str, local_path: str):
    """Download a file from the Files API to local disk."""
    content = client.beta.files.content(file_id)
    with open(local_path, "wb") as f:
        f.write(content.read())
    print(f"Downloaded {file_id} → {local_path}")


# Usage
generated = extract_generated_files(response)
for file_info in generated:
    download_file(client, file_info["file_id"], f"output/{file_info['filename']}")
```

---

## 6. File Management

### Listing Files

```python
# List all your uploaded files
files = client.beta.files.list()

for file in files.data:
    print(f"ID: {file.id}")
    print(f"  Filename: {file.filename}")
    print(f"  Created:  {file.created_at}")
    print()

# Pagination (if you have many files)
while files.has_more:
    files = client.beta.files.list(after=files.last_id)
    for file in files.data:
        print(f"{file.id}: {file.filename}")
```

### Getting File Metadata

```python
file_info = client.beta.files.retrieve(file_id)
print(f"ID:       {file_info.id}")
print(f"Filename: {file_info.filename}")
print(f"Created:  {file_info.created_at}")
```

### Deleting Files

```python
# Delete a single file
client.beta.files.delete(file_id)
print(f"Deleted {file_id}")

# Delete multiple files
def cleanup_files(client: anthropic.Anthropic, file_ids: list[str]):
    for fid in file_ids:
        try:
            client.beta.files.delete(fid)
            print(f"Deleted {fid}")
        except anthropic.NotFoundError:
            print(f"File {fid} already deleted")
        except anthropic.APIError as e:
            print(f"Failed to delete {fid}: {e}")
```

### File Lifecycle Best Practice

Always delete files after your analysis session:

```python
class ManagedFileSession:
    """Context manager that automatically cleans up uploaded files."""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self._file_ids: list[str] = []

    def upload(self, path: str, mime_type: str = "text/csv") -> str:
        """Upload a file and track it for cleanup."""
        import pathlib
        p = pathlib.Path(path)
        with open(path, "rb") as f:
            resp = self.client.beta.files.upload(
                file=(p.name, f, mime_type)
            )
        self._file_ids.append(resp.id)
        return resp.id

    def upload_bytes(self, data: bytes, filename: str, mime_type: str = "text/csv") -> str:
        """Upload bytes and track for cleanup."""
        import io
        resp = self.client.beta.files.upload(
            file=(filename, io.BytesIO(data), mime_type)
        )
        self._file_ids.append(resp.id)
        return resp.id

    def cleanup(self):
        for fid in self._file_ids:
            try:
                self.client.beta.files.delete(fid)
            except Exception:
                pass
        self._file_ids = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()


# Usage
client = anthropic.Anthropic()

with ManagedFileSession(client) as session:
    file_id = session.upload("data/sales.csv")
    # ... use file_id in requests ...
# Files automatically deleted on exit
```

---

## 7. Size Limits and Quotas

### Per-File Limits

| File type | Max size |
|-----------|---------|
| CSV / text documents | 32 MB |
| Images (PNG, JPEG, etc.) | 20 MB |
| PDF | 32 MB |
| Any single file | 32 MB |

### Storage Quotas

- **Storage quota**: Per-organisation limit (check your dashboard)
- **Files per organisation**: Large limit; delete unused files regularly
- **File retention**: Files persist until explicitly deleted (no automatic TTL)

### Practical Recommendations for Large Datasets

```python
import pandas as pd

# Check if CSV fits in single upload
import os
file_size_mb = os.path.getsize("big_dataset.csv") / (1024 * 1024)

if file_size_mb > 32:
    print("File too large — use chunking strategy")
    # Option 1: Sample the data
    df = pd.read_csv("big_dataset.csv", nrows=100_000)  # Sample
    csv_bytes = df.to_csv(index=False).encode()

    # Option 2: Aggregate before upload
    df = pd.read_csv("big_dataset.csv")
    df_agg = df.groupby(["month", "product"]).agg({"revenue": "sum"}).reset_index()
    csv_bytes = df_agg.to_csv(index=False).encode()

    # Option 3: Split into chunks
    chunk_size = 50_000
    for i, chunk_df in enumerate(pd.read_csv("big_dataset.csv", chunksize=chunk_size)):
        csv_bytes = chunk_df.to_csv(index=False).encode()
        # Upload and analyse each chunk separately
else:
    with open("big_dataset.csv", "rb") as f:
        uploaded = client.beta.files.upload(file=("big_dataset.csv", f, "text/csv"))
```

---

## 8. ZDR Considerations

Zero Data Retention (ZDR) means Anthropic does not store request/response content after serving the response.

### Files API and ZDR

**The Files API is NOT ZDR-eligible.** When you upload a file:

1. The file content is stored on Anthropic's servers
2. It persists until you delete it (or it's garbage collected)
3. This is inherently incompatible with Zero Data Retention

**Implications for sensitive data:**
- Do NOT upload PII, healthcare data (PHI), financial records, or proprietary data via the Files API if ZDR is required
- For ZDR workflows: send data inline in the message (small datasets) or use client-side tools (SubprocessRunner)

### What IS ZDR-Eligible

| Data path | ZDR eligible |
|-----------|:---:|
| Inline message content (text/base64) | ✅ |
| Client tool results | ✅ |
| Files API uploads | ❌ |
| Code execution container data | ❌ |
| Server tool results (web_search, web_fetch) | ✅ |

### Inline Alternative (ZDR-Safe)

```python
import base64

# Instead of Files API — embed directly (ZDR-safe, for small files)
with open("small_dataset.csv", "rb") as f:
    data = f.read()
    b64_data = base64.standard_b64encode(data).decode("utf-8")

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyse this CSV:"},
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "text/csv",
                    "data": b64_data,
                }
            }
        ]
    }]
)
```

---

## 9. Integration with Analysis Session

A real `AnalysisSession` class that tracks file IDs and integrates with the code execution workflow:

```python
import anthropic
import io
import time
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class UploadedFile:
    file_id: str
    filename: str
    size_bytes: int
    uploaded_at: float = field(default_factory=time.time)


class AnalysisSession:
    """
    Manages an analysis session with file tracking and lifecycle management.
    Keeps track of uploaded file_ids for re-use and cleanup.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.messages: list[dict] = []
        self.uploaded_files: list[UploadedFile] = []
        self._tools = [{"type": "code_execution_20260120", "name": "code_execution"}]

    # --- File management ---

    def upload_dataframe(self, df: pd.DataFrame, filename: str) -> str:
        """Upload a pandas DataFrame as CSV."""
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        resp = self.client.beta.files.upload(
            file=(filename, io.BytesIO(csv_bytes), "text/csv")
        )
        self.uploaded_files.append(UploadedFile(
            file_id=resp.id,
            filename=filename,
            size_bytes=len(csv_bytes)
        ))
        print(f"Uploaded DataFrame '{filename}' → {resp.id} ({len(csv_bytes):,} bytes)")
        return resp.id

    def upload_local_file(self, path: str) -> str:
        """Upload a file from disk."""
        import pathlib, os
        p = pathlib.Path(path)
        ext_to_mime = {
            ".csv": "text/csv", ".txt": "text/plain",
            ".json": "application/json", ".png": "image/png",
            ".jpg": "image/jpeg",
        }
        mime = ext_to_mime.get(p.suffix.lower(), "application/octet-stream")
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            resp = self.client.beta.files.upload(file=(p.name, f, mime))
        self.uploaded_files.append(UploadedFile(
            file_id=resp.id, filename=p.name, size_bytes=size
        ))
        return resp.id

    def get_file_id(self, filename: str) -> str | None:
        """Look up a previously uploaded file by filename."""
        for uf in self.uploaded_files:
            if uf.filename == filename:
                return uf.file_id
        return None

    # --- Conversation ---

    def ask(self, question: str, files_to_attach: list[str] = None) -> str:
        """
        Send a message, optionally attaching previously-uploaded files.
        files_to_attach: list of filenames (must be already uploaded).
        """
        content = [{"type": "text", "text": question}]

        for filename in (files_to_attach or []):
            fid = self.get_file_id(filename)
            if fid:
                content.append({
                    "type": "container_upload",
                    "file_id": fid,
                    "filename": filename
                })
            else:
                print(f"Warning: file '{filename}' not in session — skipping")

        self.messages.append({"role": "user", "content": content})
        return self._run_loop()

    def _run_loop(self) -> str:
        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                tools=self._tools,
                messages=self.messages,
                betas=["files-api-2025-04-14"],
            )
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                text_blocks = [b.text for b in response.content if b.type == "text"]
                return "\n".join(text_blocks)
            elif response.stop_reason == "pause_turn":
                self.messages.append({"role": "user", "content": "Please continue."})
            else:
                raise ValueError(f"Unexpected stop_reason: {response.stop_reason}")

    # --- Cleanup ---

    def cleanup(self):
        for uf in self.uploaded_files:
            try:
                self.client.beta.files.delete(uf.file_id)
                print(f"Deleted {uf.file_id} ({uf.filename})")
            except Exception as e:
                print(f"Could not delete {uf.file_id}: {e}")
        self.uploaded_files = []

    def __enter__(self): return self
    def __exit__(self, *args): self.cleanup()
```

---

## 10. Complete Workflow Example

Upload CSV → code execution analysis → download generated figure:

```python
import anthropic
import pandas as pd
import io
import base64
import time

def full_files_workflow():
    client = anthropic.Anthropic()

    # --- Step 1: Generate and upload data ---
    print("Step 1: Generating and uploading data...")
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=365, freq="D"),
        "temperature": [20 + 10 * __import__("math").sin(i / 365 * 2 * 3.14159)
                        + __import__("random").gauss(0, 2)
                        for i in range(365)],
        "humidity": [60 + 20 * __import__("math").cos(i / 365 * 2 * 3.14159)
                     + __import__("random").gauss(0, 5)
                     for i in range(365)],
    })

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    uploaded = client.beta.files.upload(
        file=("weather_2025.csv", io.BytesIO(csv_bytes), "text/csv")
    )
    print(f"  Uploaded: {uploaded.id}")

    # --- Step 2: Request analysis and figure generation ---
    print("Step 2: Requesting analysis...")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        tools=[{"type": "code_execution_20260120", "name": "code_execution"}],
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Load weather_2025.csv and:\n"
                        "1. Compute monthly averages for temperature and humidity\n"
                        "2. Calculate the correlation between temperature and humidity\n"
                        "3. Create a dual-axis plot: temperature (blue line) and "
                        "humidity (red dashed line) over time\n"
                        "4. Summarise the seasonal patterns you observe"
                    )
                },
                {
                    "type": "container_upload",
                    "file_id": uploaded.id,
                    "filename": "weather_2025.csv"
                }
            ]
        }],
        betas=["files-api-2025-04-14"]
    )

    # --- Step 3: Extract and display results ---
    print("Step 3: Processing results...")
    for block in response.content:
        if block.type == "text":
            print("\n" + "="*60)
            print("ANALYSIS SUMMARY")
            print("="*60)
            print(block.text)

    print(f"\nStop reason: {response.stop_reason}")
    print(f"Tokens: {response.usage.input_tokens} in + {response.usage.output_tokens} out")

    # --- Step 4: Cleanup ---
    print("Step 4: Cleaning up...")
    client.beta.files.delete(uploaded.id)
    print(f"  Deleted {uploaded.id}")

    return response


if __name__ == "__main__":
    full_files_workflow()
```
