"""Manage files in a Modal Volume from your terminal; no GPU is started.

Examples:
    python cloud_files.py init
    python cloud_files.py upload sample_data /datasets/sample_data
    python cloud_files.py list /datasets
    python cloud_files.py download /datasets/sample_data local_results
    python cloud_files.py edit /notes.txt --old draft --new ready
    python cloud_files.py delete /datasets/sample_data --recursive

Use --volume NAME before the subcommand to select another Volume. Authenticate
with Modal first; .env files are not loaded automatically. Upload checks inspect
filenames, not file contents or archives: review your data before uploading it.
Edits replace every matching occurrence in one existing UTF-8 file. Do not edit
the same cloud file concurrently: downloading and uploading is not an atomic edit.
"""

import argparse
import fnmatch
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile


DEFAULT_VOLUME = "myssp-file-demo"


def normalize_remote_path(value: str, *, allow_root: bool = True) -> str:
    """Use exact Volume paths, never parent traversal or wildcard patterns."""
    if not value or any(ord(char) < 32 for char in value):
        raise ValueError("Supply a non-empty remote path without control characters.")
    if "\\" in value or any(char in value for char in "*?[]"):
        raise ValueError("Remote paths must use / and cannot contain wildcards.")
    if ".." in value.split("/"):
        raise ValueError("Parent traversal (..) is not allowed in remote paths.")
    parts = [part for part in value.split("/") if part not in ("", ".")]
    normalized = "/" + "/".join(parts)
    if normalized == "/" and not allow_root:
        raise ValueError("The Volume root cannot be edited or deleted.")
    return normalized


def sensitive_name(name: str) -> bool:
    """Recognize common credential paths without reading their contents."""
    name = name.lower()
    if name == ".env.example":
        return False
    exact = {
        ".env", ".envrc", ".modal.toml", ".modal", ".git", ".git-credentials",
        ".netrc", ".ssh", ".aws", ".azure", ".huggingface", "gcloud",
        ".npmrc", ".pypirc", "secrets", "credentials", "token", "tokens",
        "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "private_key",
    }
    patterns = (
        ".env.*", "*.env", ".modal.toml*", "*.pem", "*.key", "*.p12", "*.pfx", "*.keystore",
        "credentials.*", "credentials*.json", "service-account*.json", "service_account*.json",
        "secrets.*", "modal_gpu_quick_start*.pdf",
    )
    return name in exact or any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _check_local_node(path: Path, *, check_secrets: bool) -> None:
    if check_secrets and sensitive_name(path.name):
        raise ValueError(f"Refusing a credential/private-guide path: {path}")
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise ValueError(f"Symlinks are not allowed: {path}")
    if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
        raise ValueError(f"Only regular files and directories are supported: {path}")


def validate_upload_path(value: str | Path) -> Path:
    """Reject an entire upload if its path/tree includes secrets or symlinks."""
    path = Path(value).expanduser().absolute()
    for parent in reversed(path.parents):
        _check_local_node(parent, check_secrets=True)
    _check_local_node(path, check_secrets=True)
    if path.is_dir():
        def walk_error(error: OSError) -> None:
            raise error

        for root, folders, files in os.walk(path, followlinks=False, onerror=walk_error):
            for name in folders + files:
                _check_local_node(Path(root) / name, check_secrets=True)
    return path


def run_modal(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    """Use this Python environment's Modal installation, without a shell."""
    return subprocess.run(
        [sys.executable, "-m", "modal", "volume", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def remote_kind(volume: str, remote: str) -> str:
    """Check each path component so edit/delete cannot traverse cloud symlinks."""
    parent = "/"
    for part in PurePosixPath(remote).parts[1:]:
        target = parent.rstrip("/") + "/" + part
        listing = run_modal("ls", "--json", volume, parent, capture=True)
        entries = json.loads(listing.stdout)
        kind = None
        for entry in entries:
            # Modal JSON keys changed from display labels to snake_case across releases.
            filename = entry.get("filename", entry.get("Filename", ""))
            # Modal escapes non-ASCII characters in the filename column.
            if "\\" in filename:
                filename = filename.encode("ascii").decode("unicode_escape")
            if "/" + filename.lstrip("/") == target:
                kind = entry.get("type", entry.get("Type", ""))
                break
        if kind is None:
            raise ValueError(f"No existing file or folder at {volume}:{target}")
        if target == remote:
            return kind
        if kind != "dir":
            raise ValueError(f"Remote parent is not a regular directory: {volume}:{target}")
        parent = target
    raise ValueError("Supply an exact file or folder below the Volume root.")


def upload(volume: str, local: str, remote: str, *, overwrite: bool = False) -> None:
    source = validate_upload_path(local)
    target = normalize_remote_path(remote)
    # A final slash means 'put this item inside that folder', as in Modal's CLI.
    if remote.endswith("/") and not target.endswith("/"):
        target += "/"
    options = ["--force"] if overwrite else []
    run_modal("put", *options, volume, str(source), target)


def download(volume: str, remote: str, local_dir: str, *, overwrite: bool = False) -> None:
    target = normalize_remote_path(remote)
    destination = Path(local_dir).expanduser().absolute()
    for path in [*reversed(destination.parents), destination]:
        if path.exists() or path.is_symlink():
            _check_local_node(path, check_secrets=False)
    if destination.exists() and not destination.is_dir():
        raise ValueError("The download destination must be a directory.")
    # Do not follow existing destination symlinks when downloading a folder.
    if destination.is_dir():
        for root, folders, files in os.walk(destination, followlinks=False):
            for name in folders + files:
                _check_local_node(Path(root) / name, check_secrets=False)
    destination.mkdir(parents=True, exist_ok=True)
    options = ["--force"] if overwrite else []
    run_modal("get", *options, volume, target, str(destination))


def edit(volume: str, remote: str, old: str, new: str) -> None:
    target = normalize_remote_path(remote, allow_root=False)
    if not old:
        raise ValueError("--old must not be empty.")
    if any(sensitive_name(part) for part in PurePosixPath(target).parts):
        raise ValueError("This demo does not edit credential/private-guide files.")
    if remote_kind(volume, target) != "file":
        raise ValueError("Only an existing regular UTF-8 text file can be edited.")
    with tempfile.TemporaryDirectory(prefix="myssp-edit-") as temporary:
        run_modal("get", volume, target, temporary)
        local_file = Path(temporary) / PurePosixPath(target).name
        if not local_file.is_file() or local_file.is_symlink():
            raise ValueError("The downloaded item is not a regular file.")
        try:
            original = local_file.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("The file is not valid UTF-8 text.") from None
        if "\x00" in original:
            raise ValueError("The file contains binary data, not plain text.")
        count = original.count(old)
        if count == 0:
            raise ValueError("The requested --old text was not found; nothing was uploaded.")
        if old == new:
            print("No change requested; nothing was uploaded.")
            return
        local_file.write_bytes(original.replace(old, new).encode("utf-8"))
        run_modal("put", "--force", volume, str(local_file), target)
    print(f"Updated {volume}:{target} ({count} replacement(s)).")


def delete(volume: str, remote: str, *, recursive: bool = False, yes: bool = False) -> None:
    target = normalize_remote_path(remote, allow_root=False)
    kind = remote_kind(volume, target)
    if kind == "dir" and not recursive:
        raise ValueError("Deleting a folder requires --recursive, including an empty folder.")
    if kind not in ("file", "dir"):
        raise ValueError("This demo deletes only regular files or folders.")
    print(f"Delete from Volume: {volume}\nExact remote path: {target}")
    if kind == "dir":
        print("All files and subfolders inside this path will be deleted.")
    if not yes and input("Type yes to delete: ").strip().lower() != "yes":
        print("Cancelled; nothing was deleted.")
        return
    options = ["--recursive"] if recursive else []
    run_modal("rm", *options, volume, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volume", default=DEFAULT_VOLUME, help="Volume name (before the subcommand)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create the named Volume with version 2 (run once)")
    listing = commands.add_parser("list", help="List a cloud folder or file")
    listing.add_argument("remote", nargs="?", default="/")
    putting = commands.add_parser("upload", help="Upload a local file or folder")
    putting.add_argument("local")
    putting.add_argument("remote", help="Exact destination; end with / to keep the local basename")
    putting.add_argument("--overwrite", action="store_true", help="Allow replacing existing cloud files")
    getting = commands.add_parser("download", help="Download a file/folder into a local directory")
    getting.add_argument("remote")
    getting.add_argument("local_dir", help="Local destination directory (created if needed)")
    getting.add_argument("--overwrite", action="store_true", help="Allow replacing existing local files")
    editing = commands.add_parser("edit", help="Replace text in one existing UTF-8 cloud file")
    editing.add_argument("remote")
    editing.add_argument("--old", required=True)
    editing.add_argument("--new", required=True)
    removing = commands.add_parser("delete", help="Delete an exact cloud file/folder after confirmation")
    removing.add_argument("remote")
    removing.add_argument("--recursive", action="store_true", help="Required to delete a folder and its contents")
    removing.add_argument("--yes", action="store_true", help="Skip the deletion confirmation")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.volume):
        parser.error("Use a Volume name containing letters, numbers, dots, dashes, or underscores.")
    try:
        if args.command == "init":
            run_modal("create", "--version=2", args.volume)
        elif args.command == "list":
            run_modal("ls", args.volume, normalize_remote_path(args.remote))
        elif args.command == "upload":
            upload(args.volume, args.local, args.remote, overwrite=args.overwrite)
        elif args.command == "download":
            download(args.volume, args.remote, args.local_dir, overwrite=args.overwrite)
        elif args.command == "edit":
            edit(args.volume, args.remote, args.old, args.new)
        elif args.command == "delete":
            delete(args.volume, args.remote, recursive=args.recursive, yes=args.yes)
    except subprocess.CalledProcessError as error:
        print(f"Modal command failed (exit {error.returncode}); see the error above.", file=sys.stderr)
        return 1
    except (ValueError, OSError, EOFError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
