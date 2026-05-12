"""
Lesson 06 — Files API (Beta)
==============================
Topics covered:
  • Uploading files (PDF, images, plain text)
  • Referencing a file_id in Messages requests
  • Listing, retrieving metadata, and deleting files
  • Supported content block types per file type
  • Storage limits and lifecycle

NOTE: The Files API is in beta — include the beta header shown below.
      It is NOT available on Amazon Bedrock or Google Vertex AI.
"""

import pathlib
import tempfile
import anthropic

# Files API requires the beta header
client = anthropic.Anthropic(
    default_headers={"anthropic-beta": "files-api-2025-04-14"}
)

MODEL = "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Helper: create a temporary text file for demo purposes
# ---------------------------------------------------------------------------
def make_temp_text_file(content: str, suffix: str = ".txt") -> pathlib.Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return pathlib.Path(tmp.name)


# ---------------------------------------------------------------------------
# 1. Upload a plain text file
# ---------------------------------------------------------------------------
def upload_text_file() -> str:
    """Upload a text document and return its file_id."""
    doc_path = make_temp_text_file(
        "The Python programming language was created by Guido van Rossum "
        "and first released in 1991. It emphasises code readability and "
        "uses significant indentation."
    )

    with open(doc_path, "rb") as f:
        response = client.beta.files.upload(
            file=(doc_path.name, f, "text/plain"),
        )

    doc_path.unlink(missing_ok=True)
    print(f"=== Uploaded text file ===\nfile_id: {response.id}\n")
    return response.id


# ---------------------------------------------------------------------------
# 2. Reference an uploaded file in a message
# ---------------------------------------------------------------------------
def query_file(file_id: str) -> None:
    """Use file_id to attach the file to a Messages request."""
    message = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    # Document content block — references the uploaded file
                    {
                        "type": "document",
                        "source": {
                            "type": "file",
                            "file_id": file_id,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Summarise the document in one sentence.",
                    },
                ],
            }
        ],
    )

    print("=== Query File ===")
    print(message.content[0].text)
    print()


# ---------------------------------------------------------------------------
# 3. Upload an image and use it in a vision request
# ---------------------------------------------------------------------------
def upload_and_query_image() -> None:
    """
    Demonstrates the image content block with file_id.
    Here we create a tiny 1×1 white PNG programmatically.
    In practice you'd open a real image file.
    """
    # Minimal valid PNG bytes (1×1 white pixel)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    response = client.beta.files.upload(
        file=("demo.png", png_bytes, "image/png"),
    )
    file_id = response.id
    print(f"=== Uploaded image ===\nfile_id: {file_id}")

    message = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "file",
                            "file_id": file_id,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe what you see in this image.",
                    },
                ],
            }
        ],
    )
    print("Claude:", message.content[0].text)

    # Clean up — delete the file when done
    client.beta.files.delete(file_id)
    print("(Image file deleted)\n")


# ---------------------------------------------------------------------------
# 4. List files in the workspace
# ---------------------------------------------------------------------------
def list_files() -> None:
    print("=== File List ===")
    files = client.beta.files.list()
    for f in files.data:
        print(f"  {f.id}  {getattr(f, 'filename', 'n/a')}  "
              f"size={getattr(f, 'size', 'n/a')} bytes")
    print()


# ---------------------------------------------------------------------------
# 5. Delete a file
# ---------------------------------------------------------------------------
def delete_file(file_id: str) -> None:
    client.beta.files.delete(file_id)
    print(f"Deleted file: {file_id}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Upload a text file, query it, then clean up
    fid = upload_text_file()
    query_file(fid)
    list_files()
    delete_file(fid)

    # Vision example
    upload_and_query_image()

    print("Files API demo complete.")
