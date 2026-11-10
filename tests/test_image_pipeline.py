"""Offline tests: no Modal hydration, network requests, or GPU work."""

from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

READY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(READY))
import image_pipeline as pipeline
import prepare_demo_data as sample


class DirectoryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, content=b"image"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path


class SelectionTests(DirectoryTest):
    def test_only_visible_image_paths_selected(self):
        expected = [self.write("dataset/a.JPG"), self.write("dataset/nested/b.png")]
        for name in ["notes.txt", ".env", ".hidden.jpg", ".git/secret.jpg", ".cache/x.png"]:
            self.write("dataset/" + name, b"placeholder-secret")
        selected = pipeline.select_images(str(self.root / "dataset"), 8)
        self.assertEqual(selected, expected)

    def test_deterministic_max_limit(self):
        for name in ["c.jpg", "a.png", "b.jpeg"]:
            self.write("dataset/" + name)
        with redirect_stdout(StringIO()):
            selected = pipeline.select_images(str(self.root / "dataset"), 2)
        self.assertEqual([p.name for p in selected], ["a.png", "b.jpeg"])

    def test_empty_folder_and_invalid_max_fail(self):
        folder = self.root / "empty"
        folder.mkdir()
        for limit in (8, 0, -1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                pipeline.select_images(str(folder), limit)

    def test_file_and_directory_symlinks_rejected(self):
        source = self.write("outside/private.jpg")
        folder = self.root / "dataset"
        folder.mkdir()
        for is_directory in (False, True):
            link = folder / ("linked-dir" if is_directory else "linked.jpg")
            link.symlink_to(source.parent if is_directory else source, target_is_directory=is_directory)
            with self.subTest(is_directory=is_directory), self.assertRaises(ValueError):
                pipeline.select_images(str(folder), 8)
            link.unlink()
        root_link = self.root / "linked-root"
        root_link.symlink_to(source.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            pipeline.select_images(str(root_link), 8)

    def test_oversized_file_rejected(self):
        path = self.write("dataset/large.jpg")
        with path.open("r+b") as file:
            file.truncate(20 * 1024 * 1024 + 1)
        with self.assertRaises(ValueError):
            pipeline.select_images(str(path.parent), 8)


class UploadTests(DirectoryTest):
    def test_unique_paths_and_manifest_available_at_batch_exit(self):
        first = self.write("dataset/one/photo.JPG", b"one")
        second = self.write("dataset/two/photo.jpg", b"two")
        captured = {}
        handles = []

        class Batch:
            def __init__(self):
                self.pending = []

            def __enter__(self):
                return self

            def put_file(self, source, target):
                self.pending.append((source, target))

            def __exit__(self, exc_type, value, traceback):
                for source, target in self.pending:
                    if isinstance(source, BytesIO):
                        self_test.assertFalse(source.closed)
                        handles.append(source)
                        captured[target] = source.read()
                    else:
                        captured[target] = Path(source).read_bytes()

        self_test = self
        volume = Mock()
        volume.batch_upload.side_effect = Batch
        with patch.object(pipeline, "volume", volume):
            pipeline.upload_images([first, second], str(self.root / "dataset"), "example-run")
        self.assertEqual(set(captured), {
            "/runs/example-run/inputs/0001.jpg",
            "/runs/example-run/inputs/0002.jpg",
            "/runs/example-run/inputs/manifest.json",
        })
        self.assertEqual(json.loads(captured["/runs/example-run/inputs/manifest.json"]), [
            {"source": "one/photo.JPG", "stored": "0001.jpg"},
            {"source": "two/photo.jpg", "stored": "0002.jpg"},
        ])
        self.assertTrue(handles[0].closed)


class DownloadTests(DirectoryTest):
    def test_streams_files_and_preserves_subdirectories(self):
        chunks = [b"first", b"", b"second"]
        volume = Mock()
        volume.read_file.side_effect = lambda path: iter(chunks)
        destination = self.root / "download"
        with patch.object(pipeline, "volume", volume):
            pipeline.download_outputs("run", ["results.json", "images/one.jpg"], destination)
        self.assertEqual((destination / "images/one.jpg").read_bytes(), b"firstsecond")
        self.assertEqual((destination / "results.json").read_bytes(), b"firstsecond")
        self.assertFalse(list(destination.rglob("*.part")))
        self.assertEqual([c.args[0] for c in volume.read_file.call_args_list], [
            "/runs/run/outputs/results.json", "/runs/run/outputs/images/one.jpg",
        ])

    def test_unsafe_paths_rejected_before_read(self):
        names = [
            "../escape.txt", "/absolute.txt", "images/../../escape.jpg",
            "x\\escape.jpg", "", ".", ".hidden", "images/.hidden.jpg",
            "images/./a.jpg", "images//a.jpg", "line\nname.jpg", "nul\0name.jpg",
        ]
        for i, name in enumerate(names):
            volume = Mock()
            with self.subTest(name=name), patch.object(pipeline, "volume", volume), self.assertRaises(ValueError):
                pipeline.download_outputs("run", [name], self.root / str(i))
            volume.read_file.assert_not_called()

    def test_windows_drive_paths_rejected(self):
        for i, name in enumerate(["C:/escape.txt", "C:escape.txt"]):
            volume = Mock()
            volume.read_file.return_value = iter([b"unexpected"])
            with self.subTest(name=name), patch.object(pipeline, "volume", volume), self.assertRaises(ValueError):
                pipeline.download_outputs("run", [name], self.root / str(i))
            volume.read_file.assert_not_called()

    def test_failed_stream_removes_partial_and_has_no_final_file(self):
        def failing_stream(path):
            yield b"partial"
            raise OSError("simulated interrupted download")

        volume = Mock()
        volume.read_file.side_effect = failing_stream
        destination = self.root / "download"
        with patch.object(pipeline, "volume", volume), self.assertRaises(OSError):
            pipeline.download_outputs("run", ["images/a.jpg"], destination)
        self.assertFalse((destination / "images/a.jpg").exists())
        self.assertFalse((destination / "images/a.jpg.part").exists())

    def test_existing_destination_preserved(self):
        existing = self.write("download/results.json", b"keep me")
        volume = Mock()
        with patch.object(pipeline, "volume", volume), self.assertRaises(FileExistsError):
            pipeline.download_outputs("run", ["results.json"], existing.parent)
        self.assertEqual(existing.read_bytes(), b"keep me")
        volume.read_file.assert_not_called()

    def test_duplicate_and_canonical_alias_paths_rejected(self):
        for i, names in enumerate([
            ["results.json", "results.json"],
            ["images/a.jpg", "images/./a.jpg"],
        ]):
            volume = Mock()
            volume.read_file.side_effect = lambda path: iter([b"content"])
            with self.subTest(names=names), patch.object(pipeline, "volume", volume), self.assertRaises((ValueError, FileExistsError)):
                pipeline.download_outputs("run", names, self.root / str(i))


class SampleDataTests(DirectoryTest):
    def run_main(self, folder, response=None):
        with patch.object(sys, "argv", ["prepare_demo_data.py", "--output-dir", str(folder)]), \
             patch.object(sample, "urlopen", return_value=response) as fetch, \
             redirect_stdout(StringIO()):
            sample.main()
        return fetch

    def test_valid_response_saved_once(self):
        data = b"\xff\xd8\xffmock-jpeg-data"
        folder = self.root / "sample"
        fetch = self.run_main(folder, BytesIO(data))
        self.assertEqual((folder / "dog.jpg").read_bytes(), data)
        fetch.assert_called_once_with(sample.SOURCE_URL, timeout=30)

    def test_invalid_or_oversized_response_not_saved(self):
        for i, data in enumerate([b"<html>not an image</html>", b"\xff\xd8\xff" + b"x" * (5 * 1024 * 1024)]):
            folder = self.root / str(i)
            with self.subTest(i=i), self.assertRaises(ValueError):
                self.run_main(folder, BytesIO(data))
            self.assertFalse((folder / "dog.jpg").exists())

    def test_existing_file_not_downloaded_or_overwritten(self):
        existing = self.write("sample/dog.jpg", b"keep me")
        with patch.object(sys, "argv", ["prepare_demo_data.py", "--output-dir", str(existing.parent)]), \
             patch.object(sample, "urlopen") as fetch, \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            sample.main()
        fetch.assert_not_called()
        self.assertEqual(existing.read_bytes(), b"keep me")


if __name__ == "__main__":
    unittest.main()
