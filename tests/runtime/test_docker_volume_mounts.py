"""Tests for Docker volume mounting functionality in docker_runtime.py."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from strix.runtime import SandboxInitializationError
from strix.runtime.docker_runtime import DockerRuntime


class TestPrepareVolumeMounts:
    """Tests for the _prepare_volume_mounts method."""

    @pytest.fixture
    def docker_runtime(self) -> DockerRuntime:
        """Create a DockerRuntime instance with mocked client."""
        with patch("strix.runtime.docker_runtime.docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            runtime = DockerRuntime()
            return runtime

    def test_no_local_sources_returns_none(self, docker_runtime: DockerRuntime) -> None:
        """Test that None is returned when no local sources are provided."""
        result = docker_runtime._prepare_volume_mounts(None)
        assert result is None

    def test_empty_local_sources_returns_none(self, docker_runtime: DockerRuntime) -> None:
        """Test that None is returned when empty local sources list is provided."""
        result = docker_runtime._prepare_volume_mounts([])
        assert result is None

    def test_single_valid_directory(self, docker_runtime: DockerRuntime) -> None:
        """Test mounting a single valid directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_sources = [{"source_path": tmpdir}]
            result = docker_runtime._prepare_volume_mounts(local_sources)

            assert result is not None
            assert len(result) == 1
            # Result should have the resolved path as key
            resolved_path = str(Path(tmpdir).resolve())
            assert resolved_path in result
            assert result[resolved_path]["bind"] == f"/workspace/{Path(tmpdir).name}"
            assert result[resolved_path]["mode"] == "rw"

    def test_directory_with_workspace_subdir(self, docker_runtime: DockerRuntime) -> None:
        """Test mounting with custom workspace_subdir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_sources = [{"source_path": tmpdir, "workspace_subdir": "custom_dir"}]
            result = docker_runtime._prepare_volume_mounts(local_sources)

            assert result is not None
            resolved_path = str(Path(tmpdir).resolve())
            assert result[resolved_path]["bind"] == "/workspace/custom_dir"

    def test_multiple_directories(self, docker_runtime: DockerRuntime) -> None:
        """Test mounting multiple directories."""
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            local_sources = [
                {"source_path": tmpdir1, "workspace_subdir": "dir1"},
                {"source_path": tmpdir2, "workspace_subdir": "dir2"},
            ]
            result = docker_runtime._prepare_volume_mounts(local_sources)

            assert result is not None
            assert len(result) == 2

            resolved_path1 = str(Path(tmpdir1).resolve())
            resolved_path2 = str(Path(tmpdir2).resolve())
            assert resolved_path1 in result
            assert resolved_path2 in result
            assert result[resolved_path1]["bind"] == "/workspace/dir1"
            assert result[resolved_path2]["bind"] == "/workspace/dir2"

    def test_nonexistent_path_raises_error(self, docker_runtime: DockerRuntime) -> None:
        """Test that non-existent path raises SandboxInitializationError."""
        local_sources = [{"source_path": "/nonexistent/path/that/does/not/exist"}]

        with pytest.raises(SandboxInitializationError) as exc_info:
            docker_runtime._prepare_volume_mounts(local_sources)

        assert "Source path does not exist" in str(exc_info.value.message)
        assert "/nonexistent/path/that/does/not/exist" in str(exc_info.value.details)

    def test_file_instead_of_directory_raises_error(
        self, docker_runtime: DockerRuntime
    ) -> None:
        """Test that a file path instead of directory raises SandboxInitializationError."""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(b"test content")
            tmpfile_path = tmpfile.name

        try:
            local_sources = [{"source_path": tmpfile_path}]

            with pytest.raises(SandboxInitializationError) as exc_info:
                docker_runtime._prepare_volume_mounts(local_sources)

            assert "Source path is not a directory" in str(exc_info.value.message)
        finally:
            Path(tmpfile_path).unlink()

    def test_source_without_path_is_skipped(self, docker_runtime: DockerRuntime) -> None:
        """Test that sources without source_path are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_sources = [
                {"workspace_subdir": "no_path"},  # Missing source_path
                {"source_path": tmpdir},  # Valid source
            ]
            result = docker_runtime._prepare_volume_mounts(local_sources)

            assert result is not None
            assert len(result) == 1  # Only the valid source should be included

    def test_source_with_empty_path_is_skipped(self, docker_runtime: DockerRuntime) -> None:
        """Test that sources with empty source_path are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_sources = [
                {"source_path": ""},  # Empty source_path
                {"source_path": tmpdir},  # Valid source
            ]
            result = docker_runtime._prepare_volume_mounts(local_sources)

            assert result is not None
            assert len(result) == 1  # Only the valid source should be included

    def test_relative_path_is_resolved(self, docker_runtime: DockerRuntime) -> None:
        """Test that relative paths are resolved to absolute paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a subdirectory and navigate to parent
            import os

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                subdir = Path("test_subdir")
                subdir.mkdir()

                local_sources = [{"source_path": "./test_subdir"}]
                result = docker_runtime._prepare_volume_mounts(local_sources)

                assert result is not None
                # Should have an absolute path as key, not relative
                for path in result.keys():
                    assert Path(path).is_absolute()
                    assert "test_subdir" in path
            finally:
                os.chdir(original_cwd)

    def test_fallback_target_name_when_no_workspace_subdir(
        self, docker_runtime: DockerRuntime
    ) -> None:
        """Test fallback to directory name when workspace_subdir is not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_name = Path(tmpdir).name
            local_sources = [{"source_path": tmpdir}]
            result = docker_runtime._prepare_volume_mounts(local_sources)

            assert result is not None
            resolved_path = str(Path(tmpdir).resolve())
            assert result[resolved_path]["bind"] == f"/workspace/{dir_name}"


class TestVolumesMatch:
    """Tests for the _volumes_match method."""

    @pytest.fixture
    def docker_runtime(self) -> DockerRuntime:
        """Create a DockerRuntime instance with mocked client."""
        with patch("strix.runtime.docker_runtime.docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            runtime = DockerRuntime()
            return runtime

    def _make_container(self, binds: list[str] | None = None) -> MagicMock:
        """Create a mock container with the given bind mounts."""
        container = MagicMock()
        container.attrs = {
            "HostConfig": {"Binds": binds or []},
        }
        return container

    def test_both_none_matches(self, docker_runtime: DockerRuntime) -> None:
        """No existing binds and no requested volumes should match."""
        container = self._make_container(binds=None)
        assert docker_runtime._volumes_match(container, None) is True

    def test_both_empty_matches(self, docker_runtime: DockerRuntime) -> None:
        """Empty binds and empty requested volumes should match."""
        container = self._make_container(binds=[])
        assert docker_runtime._volumes_match(container, {}) is True

    def test_existing_binds_no_requested_mismatch(self, docker_runtime: DockerRuntime) -> None:
        """Container has binds but no volumes requested -> mismatch."""
        container = self._make_container(binds=["/host/path:/container/path:rw"])
        assert docker_runtime._volumes_match(container, None) is False

    def test_no_existing_binds_requested_mismatch(self, docker_runtime: DockerRuntime) -> None:
        """Container has no binds but volumes are requested -> mismatch."""
        container = self._make_container(binds=[])
        volumes = {"/host/path": {"bind": "/container/path", "mode": "rw"}}
        assert docker_runtime._volumes_match(container, volumes) is False

    def test_matching_volumes(self, docker_runtime: DockerRuntime) -> None:
        """Container binds match requested volumes exactly."""
        container = self._make_container(binds=["/host/path:/workspace/app:rw"])
        volumes = {"/host/path": {"bind": "/workspace/app", "mode": "rw"}}
        assert docker_runtime._volumes_match(container, volumes) is True

    def test_different_host_path_mismatch(self, docker_runtime: DockerRuntime) -> None:
        """Different host paths should not match."""
        container = self._make_container(binds=["/old/path:/workspace/app:rw"])
        volumes = {"/new/path": {"bind": "/workspace/app", "mode": "rw"}}
        assert docker_runtime._volumes_match(container, volumes) is False

    def test_different_container_path_mismatch(self, docker_runtime: DockerRuntime) -> None:
        """Different container paths should not match."""
        container = self._make_container(binds=["/host/path:/workspace/old:rw"])
        volumes = {"/host/path": {"bind": "/workspace/new", "mode": "rw"}}
        assert docker_runtime._volumes_match(container, volumes) is False

    def test_multiple_volumes_match(self, docker_runtime: DockerRuntime) -> None:
        """Multiple matching volumes in any order."""
        container = self._make_container(
            binds=["/b:/workspace/b:rw", "/a:/workspace/a:rw"]
        )
        volumes = {
            "/a": {"bind": "/workspace/a", "mode": "rw"},
            "/b": {"bind": "/workspace/b", "mode": "rw"},
        }
        assert docker_runtime._volumes_match(container, volumes) is True


class TestExpandUser:
    """Test that expanduser is called for paths with ~."""

    @pytest.fixture
    def docker_runtime(self) -> DockerRuntime:
        with patch("strix.runtime.docker_runtime.docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            return DockerRuntime()

    def test_tilde_path_is_expanded(self, docker_runtime: DockerRuntime) -> None:
        """Test that ~ in source_path is expanded to the home directory."""
        home = str(Path.home())
        # Create a temp dir inside home to test
        test_dir = Path(home) / "strix_test_expanduser_tmp"
        test_dir.mkdir(exist_ok=True)
        try:
            local_sources = [{"source_path": "~/strix_test_expanduser_tmp"}]
            result = docker_runtime._prepare_volume_mounts(local_sources)

            assert result is not None
            resolved = str(test_dir.resolve())
            assert resolved in result
        finally:
            test_dir.rmdir()
