"""Local tests only: every Modal operation is mocked."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cloud_files", HERE.parent / "cloud_files.py")
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)


class FakeVolume:
    def __init__(self, entries=None):
        self.entries = entries or {"/notes.txt": ("file", b"draft\r\ndraft\n")}
        self.calls = []
        self.uploaded = None
        self.temporary = None

    def __call__(self, *args, capture=False):
        self.calls.append(args)
        if args[0] == "ls":
            parent = args[-1]
            listing = [
                {"filename": name.lstrip("/"), "type": kind}
                for name, (kind, _) in self.entries.items()
                if str(Path(name).parent) == parent
            ]
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(listing))
        if args[0] == "get":
            remote, directory = args[-2:]
            target = Path(directory) / Path(remote).name
            target.write_bytes(self.entries[remote][1])
            self.temporary = Path(directory)
        if args[0] == "put":
            self.uploaded = Path(args[-2]).read_bytes()
        return subprocess.CompletedProcess(args, 0)


class CloudFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="file-tests-", dir=HERE)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_remote_paths(self):
        self.assertEqual(cloud.normalize_remote_path("runs//one/./file.txt"), "/runs/one/file.txt")
        for value in ("../notes", "a/../b", "a\\b", "*", "a?", "[ab]", "", "a\nb"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cloud.normalize_remote_path(value)

    def test_root_and_parent_deletion_never_call_modal(self):
        with patch.object(cloud, "run_modal") as modal:
            for value in ("/", ".", "////", "a/../", "../notes"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    cloud.delete("test-volume", value, recursive=True, yes=True)
            modal.assert_not_called()

    def test_sensitive_nested_upload_is_rejected_without_cloud_call(self):
        (self.root / "data").mkdir()
        (self.root / "data" / ".env.local").write_text("placeholder")
        with patch.object(cloud, "run_modal") as modal, self.assertRaises(ValueError):
            cloud.upload("test-volume", str(self.root), "/data")
        modal.assert_not_called()

    def test_sensitive_names_and_template(self):
        names = (".env", ".env.local", "dev.env", ".modal.toml.bak", ".git", "id_rsa", "private.key",
                 "credentials.json", "secrets.toml", "modal_gpu_quick_start_team3.pdf")
        for name in names:
            with self.subTest(name=name):
                self.assertTrue(cloud.sensitive_name(name))
        sample = self.root / ".env.example"
        sample.write_text("PLACEHOLDER=example")
        self.assertEqual(cloud.validate_upload_path(sample), sample)

    def test_upload_rejects_file_directory_and_ancestor_symlinks(self):
        data = self.root / "data"
        data.mkdir()
        original = data / "a.txt"
        original.write_text("hello")
        file_link = self.root / "file-link"
        file_link.symlink_to(original)
        folder_link = self.root / "folder-link"
        folder_link.symlink_to(data, target_is_directory=True)
        for path in (file_link, folder_link, folder_link / "a.txt", self.root):
            with self.subTest(path=path), self.assertRaises(ValueError):
                cloud.validate_upload_path(path)

    def test_upload_overwrites_only_when_explicit(self):
        source = self.root / "notes.txt"
        source.write_text("hello")
        with patch.object(cloud, "run_modal") as modal:
            cloud.upload("test-volume", str(source), "/data/")
            modal.assert_called_once_with("put", "test-volume", str(source), "/data/")
            modal.reset_mock()
            cloud.upload("test-volume", str(source), "/renamed.txt", overwrite=True)
            modal.assert_called_once_with("put", "--force", "test-volume", str(source), "/renamed.txt")

    def test_download_precreates_directory_and_preserves_overwrite_default(self):
        destination = self.root / "recovered"
        with patch.object(cloud, "run_modal") as modal:
            cloud.download("test-volume", "runs/a/outputs", str(destination))
            self.assertTrue(destination.is_dir())
            modal.assert_called_once_with("get", "test-volume", "/runs/a/outputs", str(destination))

    def test_download_rejects_existing_symlink(self):
        destination = self.root / "recovered"
        destination.mkdir()
        (destination / "link").symlink_to(self.root)
        with patch.object(cloud, "run_modal") as modal, self.assertRaises(ValueError):
            cloud.download("test-volume", "/outputs", str(destination), overwrite=True)
        modal.assert_not_called()

    def test_edit_changes_existing_text_preserves_newlines_and_cleans_temp(self):
        fake = FakeVolume()
        with patch.object(cloud, "run_modal", side_effect=fake), contextlib.redirect_stdout(io.StringIO()) as output:
            cloud.edit("test-volume", "/notes.txt", "draft", "ready")
        self.assertEqual(fake.uploaded, b"ready\r\nready\n")
        self.assertFalse(fake.temporary.exists())
        self.assertIn("2 replacement(s)", output.getvalue())
        self.assertNotIn("ready", output.getvalue())

    def test_edit_missing_old_text_never_uploads_and_cleans_temp(self):
        fake = FakeVolume()
        with patch.object(cloud, "run_modal", side_effect=fake), self.assertRaises(ValueError):
            cloud.edit("test-volume", "/notes.txt", "missing", "replacement")
        self.assertIsNone(fake.uploaded)
        self.assertFalse(fake.temporary.exists())

    def test_edit_missing_file_or_directory_never_downloads_or_creates(self):
        for entries in ({"/other.txt": ("file", b"")}, {"/notes.txt": ("dir", b"")}):
            fake = FakeVolume(entries)
            with self.subTest(entries=entries), patch.object(cloud, "run_modal", side_effect=fake), self.assertRaises(ValueError):
                cloud.edit("test-volume", "/notes.txt", "old", "new")
            self.assertTrue(all(call[0] == "ls" for call in fake.calls))

    def test_edit_binary_text_does_not_upload(self):
        for data in (b"\xff", b"draft\x00"):
            fake = FakeVolume({"/notes.txt": ("file", data)})
            with self.subTest(data=data), patch.object(cloud, "run_modal", side_effect=fake), self.assertRaises(ValueError):
                cloud.edit("test-volume", "/notes.txt", "draft", "ready")
            self.assertIsNone(fake.uploaded)
            self.assertFalse(fake.temporary.exists())

    def test_edit_and_delete_refuse_final_or_ancestor_cloud_symlink(self):
        for remote, entries in (
            ("/notes.txt", {"/notes.txt": ("link", b"")}),
            ("/linked/notes.txt", {"/linked": ("link", b"")}),
        ):
            for operation in ("edit", "delete"):
                fake = FakeVolume(entries)
                with self.subTest(remote=remote, operation=operation), patch.object(cloud, "run_modal", side_effect=fake), self.assertRaises(ValueError):
                    if operation == "edit":
                        cloud.edit("test-volume", remote, "old", "new")
                    else:
                        cloud.delete("test-volume", remote, recursive=True, yes=True)
                self.assertTrue(all(call[0] == "ls" for call in fake.calls))

    def test_delete_confirmation_cancellation_and_exact_target_output(self):
        fake = FakeVolume()
        with patch.object(cloud, "run_modal", side_effect=fake), patch("builtins.input", return_value="no"), contextlib.redirect_stdout(io.StringIO()) as output:
            cloud.delete("test-volume", "/notes.txt")
        self.assertFalse(any(call[0] == "rm" for call in fake.calls))
        self.assertIn("test-volume", output.getvalue())
        self.assertIn("/notes.txt", output.getvalue())

    def test_folder_delete_requires_recursive_even_if_empty(self):
        fake = FakeVolume({"/folder": ("dir", b"")})
        with patch.object(cloud, "run_modal", side_effect=fake), self.assertRaises(ValueError):
            cloud.delete("test-volume", "/folder", yes=True)
        self.assertFalse(any(call[0] == "rm" for call in fake.calls))
        with patch.object(cloud, "run_modal", side_effect=fake), contextlib.redirect_stdout(io.StringIO()):
            cloud.delete("test-volume", "/folder", recursive=True, yes=True)
        self.assertEqual(fake.calls[-1], ("rm", "--recursive", "test-volume", "/folder"))

    def test_nested_file_metadata_checks_all_ancestors(self):
        fake = FakeVolume({"/run": ("dir", b""), "/run/notes.txt": ("file", b"draft")})
        with patch.object(cloud, "run_modal", side_effect=fake):
            self.assertEqual(cloud.remote_kind("test-volume", "/run/notes.txt"), "file")
        self.assertEqual(fake.calls, [("ls", "--json", "test-volume", "/"), ("ls", "--json", "test-volume", "/run")])

    def test_subprocess_uses_python_and_argument_list(self):
        with patch.object(cloud.subprocess, "run") as run:
            cloud.run_modal("ls", "test-volume", "/")
        run.assert_called_once_with([sys.executable, "-m", "modal", "volume", "ls", "test-volume", "/"], check=True, text=True, stdout=None)

    def test_init_uses_selected_volume_and_version_two(self):
        with patch.object(cloud, "run_modal") as modal:
            self.assertEqual(cloud.main(["--volume", "team-test", "init"]), 0)
        modal.assert_called_once_with("create", "--version=2", "team-test")


if __name__ == "__main__":
    unittest.main()
