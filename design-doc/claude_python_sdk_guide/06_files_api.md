# Chapter 6 — Files API

*← [Chapter 5: Message Batches](05_batch_processing.md) | [Chapter 7: Error Handling](07_error_handling.md) →*

---

## What Is the Files API?

The **Files API** lets you upload documents, images, and other files to Anthropic's secure storage and reference them repeatedly across multiple API calls — without re-uploading the content each time.

**Without Files API:**
```
Request 1: [system prompt] + [full PDF bytes] + [question]   → 50,000 tokens
Request 2: [system prompt] + [full PDF bytes] + [question]   → 50,000 tokens
Request 3: [system prompt] + [full PDF bytes] + [question]   → 50,000 tokens
                                                   Total: 150,000 input tokens
```

**With Files API:**
```
Upload:    PUT /v1/files  ← PDF bytes sent once
Request 1: [system prompt] + [file_id] + [question]          → 150 tokens
Request 2: [system prompt] + [file_id] + [question]          → 150 tokens
Request 3: [system prompt] + [file_id] + [question]          → 150 tokens
                                                   Total: 450 input tokens + file once
```

> **Note:** The Files API is in **beta**. It is not yet available on Amazon Bedrock or Google Vertex AI.

---

## Enabling the Files API

Include the beta header in the client or per-request:

```python
import anthropic

# Client-level — applies to all requests
client = anthropic.Anthropic(
    default_headers={"anthropic-beta": "files-api-2025-04-14"}
)
```

---

## Supported File Types

| File Type | MIME Type | Content Block | Use Case |
|-----------|-----------|--------------|---------|
| PDF | `application/pdf` | `document` | Document Q&A, summarisation, analysis |
| Plain text | `text/plain` | `document` | Logs, code, text analysis |
| JPEG | `image/jpeg` | `image` | Vision tasks |
| PNG | `image/png` | `image` | Vision tasks |
| GIF | `image/gif` | `image` | Vision tasks |
| WebP | `image/webp` | `image` | Vision tasks |
| CSV, XLSX, others | Varies | `container_upload` | Code execution tool |

---

## Storage Limits & Lifecycle

| Limit | Value |
|-------|-------|
| Maximum file size | 500 MB |
| Total storage per organisation | 500 GB |
| File-related API calls | ~100 requests/minute (beta) |
| File retention | Until explicitly deleted |
| Workspace scope | Files are shared across all API keys in the same workspace |

---

## Core Operations

### Upload a file

```python
import anthropic
from pathlib import Path

client = anthropic.Anthropic(
    default_headers={"anthropic-beta": "files-api-2025-04-14"}
)

# Upload a PDF
pdf_path = Path("annual_report.pdf")
with open(pdf_path, "rb") as f:
    response = client.beta.files.upload(
        file=(pdf_path.name, f, "application/pdf"),
    )

file_id = response.id
print(f"Uploaded: {file_id}")
```

The response contains:
- `id` — the `file_id` to use in future requests
- `filename` — the name you provided
- `size` — file size in bytes
- `created_at` — upload timestamp

### Upload an image

```python
from pathlib import Path

img_path = Path("product_photo.jpg")
with open(img_path, "rb") as f:
    response = client.beta.files.upload(
        file=(img_path.name, f, "image/jpeg"),
    )
image_id = response.id
```

### Upload plain text

```python
text_content = "Claude was created by Anthropic in 2021…"

response = client.beta.files.upload(
    file=("knowledge.txt", text_content.encode("utf-8"), "text/plain"),
)
text_id = response.id
```

---

## Using a File in a Message

Reference the `file_id` in a content block — the file is automatically attached:

### PDF document

```python
message = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "file",
                        "file_id": file_id,          # from upload response
                    },
                },
                {
                    "type": "text",
                    "text": "Summarise the key financial highlights from this report.",
                },
            ],
        }
    ],
)
print(message.content[0].text)
```

### Image

```python
message = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=512,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "file",
                        "file_id": image_id,
                    },
                },
                {
                    "type": "text",
                    "text": "What objects are visible in this image?",
                },
            ],
        }
    ],
)
```

### Multiple files in one request

```python
message = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": doc1_id},
                },
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": doc2_id},
                },
                {
                    "type": "text",
                    "text": "Compare and contrast the two documents.",
                },
            ],
        }
    ],
)
```

---

## Managing Files

### List all uploaded files

```python
files = client.beta.files.list()
for f in files.data:
    print(f"{f.id}  {f.filename}  {f.size} bytes")
```

### Get metadata for a specific file

```python
file_info = client.beta.files.retrieve(file_id)
print(f"Name: {file_info.filename}")
print(f"Size: {file_info.size} bytes")
print(f"Created: {file_info.created_at}")
```

### Delete a file

```python
client.beta.files.delete(file_id)
print(f"Deleted: {file_id}")
```

> **Note:** Files you uploaded cannot be downloaded back. Only files **created by** the code execution tool or skills can be downloaded.

---

## Real-World Pattern: Document Q&A System

```python
import anthropic
from pathlib import Path

client = anthropic.Anthropic(
    default_headers={"anthropic-beta": "files-api-2025-04-14"}
)

class DocumentQA:
    """Upload a document once, ask many questions."""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        self.file_id = self._upload()

    def _upload(self) -> str:
        with open(self.pdf_path, "rb") as f:
            resp = client.beta.files.upload(
                file=(self.pdf_path.name, f, "application/pdf"),
            )
        print(f"Uploaded '{self.pdf_path.name}' → {resp.id}")
        return resp.id

    def ask(self, question: str) -> str:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": self.file_id},
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )
        return resp.content[0].text

    def cleanup(self):
        client.beta.files.delete(self.file_id)
        print(f"Deleted file: {self.file_id}")


# Usage
# qa = DocumentQA("research_paper.pdf")
# print(qa.ask("What is the main hypothesis of this paper?"))
# print(qa.ask("What methods were used?"))
# print(qa.ask("What were the key findings?"))
# qa.cleanup()
```

---

## Error Reference

| Error | Cause | Fix |
|-------|-------|-----|
| `404 File not found` | `file_id` doesn't exist or wrong workspace | Check ID, verify workspace |
| `400 Invalid file type` | Using image `file_id` in a `document` block | Match file type to content block type |
| `400 Exceeds context window` | File too large for the model's context | Use a model with larger context or split the file |
| `413 File too large` | File > 500 MB | Split or compress the file |
| `403 Storage limit exceeded` | Org has used 500 GB | Delete unused files |

---

## Async File Operations

All file operations work identically with `AsyncAnthropic`:

```python
import asyncio
import anthropic

async def upload_and_query(pdf_path: str) -> str:
    client = anthropic.AsyncAnthropic(
        default_headers={"anthropic-beta": "files-api-2025-04-14"}
    )

    with open(pdf_path, "rb") as f:
        upload_resp = await client.beta.files.upload(
            file=(pdf_path, f, "application/pdf"),
        )

    message = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "file", "file_id": upload_resp.id},
                    },
                    {"type": "text", "text": "What is this document about?"},
                ],
            }
        ],
    )

    await client.beta.files.delete(upload_resp.id)
    await client.aclose()
    return message.content[0].text

# result = asyncio.run(upload_and_query("report.pdf"))
```

---

*← [Chapter 5: Message Batches](05_batch_processing.md) | [Chapter 7: Error Handling](07_error_handling.md) →*
