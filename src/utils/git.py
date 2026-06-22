"""
Git repository integration — fetch *.json files from GitHub, GitLab, or any
other git host.

Provider routing
----------------
github.com          → PyGithub SDK   (Contents API, no clone needed)
gitlab.com / gitlab.* → python-gitlab SDK (Repository Files API, no clone needed)
everything else     → git clone fallback (Bitbucket, Azure DevOps, self-hosted Gitea, etc.)
"""

import json
import shutil
import subprocess
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_json_files(
    url:    str,
    token:  str = "",
    branch: str = "main",
    subdir: str = "",
) -> tuple[bool, dict[str, Any], str]:
    """
    Fetch every *.json file from a Git repository directory.

    Returns
    -------
    (success, file_map, error_message)
      success   : bool
      file_map  : {relative_path: parsed_json}  (empty on failure)
      error_msg : human-readable error string   (empty on success)
    """
    provider = _detect_provider(url)
    log.info("fetch_json_files: url=%s branch=%s subdir=%s provider=%s", url, branch, subdir, provider)
    if provider == "github":
        return _fetch_github(url, token, branch, subdir)
    if provider == "gitlab":
        return _fetch_gitlab(url, token, branch, subdir)
    return _fetch_via_clone(url, token, branch, subdir)


def list_branches(url: str, token: str = "") -> list[str]:
    """
    Return the list of branch names for a repository.
    Returns an empty list on any error (caller should fall back to text input).
    """
    provider = _detect_provider(url)
    try:
        if provider == "github":
            from github import Github
            g    = Github(token) if token else Github()
            repo = g.get_repo(_repo_path(url))
            return [b.name for b in repo.get_branches()]
        if provider == "gitlab":
            import gitlab
            parsed  = urllib.parse.urlparse(url)
            base    = f"{parsed.scheme}://{parsed.netloc}"
            gl      = gitlab.Gitlab(base, private_token=token or None)
            project = gl.projects.get(_repo_path(url))
            return [b.name for b in project.branches.list(get_all=True)]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider(url: str) -> str:
    host = urllib.parse.urlparse(url).netloc.lower()
    if "github.com" in host:
        return "github"
    if "gitlab" in host:          # gitlab.com and self-hosted (e.g. gitlab.myco.com)
        return "gitlab"
    return "other"


def _repo_path(url: str) -> str:
    """Extract 'owner/repo' (or 'group/subgroup/repo') from a URL."""
    path = urllib.parse.urlparse(url).path.strip("/")
    return path[:-4] if path.endswith(".git") else path


# ---------------------------------------------------------------------------
# GitHub — PyGithub
# ---------------------------------------------------------------------------

def _fetch_github(
    url: str, token: str, branch: str, subdir: str
) -> tuple[bool, dict[str, Any], str]:
    try:
        from github import Github, GithubException
    except ImportError:
        return False, {}, "PyGithub not installed. Run: pip install PyGithub"

    try:
        g    = Github(token) if token else Github()
        repo = g.get_repo(_repo_path(url))
        file_map: dict[str, Any] = {}
        _github_collect(repo, subdir or "", branch, file_map)
        if not file_map:
            return False, {}, f"No *.json files found in '{subdir or '/'}' on branch '{branch}'"
        return True, file_map, ""

    except GithubException as exc:
        msgs = {
            401: "Authentication failed — check your GitHub token.",
            403: f"Access denied or rate-limited: {exc.data.get('message', '')}",
            404: f"Repository or branch not found: {url}",
        }
        return False, {}, msgs.get(exc.status, str(exc))
    except Exception as exc:
        return False, {}, str(exc)


def _github_collect(repo: Any, path: str, branch: str, result: dict) -> None:
    """Recursively collect *.json files from a GitHub repo path."""
    from github import GithubException
    try:
        items = repo.get_contents(path, ref=branch)
    except GithubException:
        return
    if not isinstance(items, list):
        items = [items]
    for item in items:
        if item.type == "dir":
            _github_collect(repo, item.path, branch, result)
        elif item.name.endswith(".json"):
            try:
                result[item.path] = json.loads(item.decoded_content.decode("utf-8"))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# GitLab — python-gitlab
# ---------------------------------------------------------------------------

def _fetch_gitlab(
    url: str, token: str, branch: str, subdir: str
) -> tuple[bool, dict[str, Any], str]:
    try:
        import gitlab
    except ImportError:
        return False, {}, "python-gitlab not installed. Run: pip install python-gitlab"

    try:
        parsed  = urllib.parse.urlparse(url)
        base    = f"{parsed.scheme}://{parsed.netloc}"
        gl      = gitlab.Gitlab(base, private_token=token or None)
        project = gl.projects.get(_repo_path(url))

        tree = project.repository_tree(
            path=subdir or "",
            ref=branch,
            recursive=True,
            get_all=True,
        )
        file_map: dict[str, Any] = {}
        for item in tree:
            if item["type"] == "blob" and item["name"].endswith(".json"):
                try:
                    f       = project.files.get(file_path=item["path"], ref=branch)
                    content = f.decode()
                    if isinstance(content, bytes):
                        content = content.decode("utf-8")
                    file_map[item["path"]] = json.loads(content)
                except Exception:
                    pass

        if not file_map:
            return False, {}, f"No *.json files found in '{subdir or '/'}' on branch '{branch}'"
        return True, file_map, ""

    except Exception as exc:
        err = str(exc)
        if token:
            err = err.replace(token, "***")
        return False, {}, err


# ---------------------------------------------------------------------------
# Fallback — git clone (Bitbucket, Azure DevOps, Gitea, etc.)
# ---------------------------------------------------------------------------

_TOKEN_PREFIX: dict[str, str] = {
    "bitbucket.org": "x-token-auth",
}


def _fetch_via_clone(
    url: str, token: str, branch: str, subdir: str
) -> tuple[bool, dict[str, Any], str]:
    """Clone the repo shallowly and load *.json files from the target subdir."""
    tmp        = tempfile.mkdtemp(prefix="json_spell_git_")
    clone_url  = url

    if token:
        parsed   = urllib.parse.urlparse(url)
        host     = parsed.netloc.lower()
        prefix   = next((v for k, v in _TOKEN_PREFIX.items() if k in host), None)
        netloc   = f"{prefix}:{token}@{host}" if prefix else f"{token}@{host}"
        clone_url = parsed._replace(netloc=netloc).geturl()

    try:
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, clone_url, tmp],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp, ignore_errors=True)
        return False, {}, "Git clone timed out (>60s). Check the URL and your network."
    except FileNotFoundError:
        shutil.rmtree(tmp, ignore_errors=True)
        return False, {}, "`git` is not installed or not on PATH."

    if proc.returncode != 0:
        err = proc.stderr.replace(token, "***") if token else proc.stderr
        shutil.rmtree(tmp, ignore_errors=True)
        return False, {}, err.strip()

    scan = Path(tmp) / subdir.lstrip("/") if subdir else Path(tmp)
    if not scan.is_dir():
        shutil.rmtree(tmp, ignore_errors=True)
        return False, {}, f"Subdirectory '{subdir}' not found in the cloned repo."

    file_map = _load_json_dir(scan)
    shutil.rmtree(tmp, ignore_errors=True)

    if not file_map:
        return False, {}, f"No *.json files found in '{subdir or '/'}'"
    return True, file_map, ""


def _load_json_dir(directory: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for p in sorted(directory.rglob("*.json")):
        try:
            result[str(p.relative_to(directory))] = json.loads(
                p.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            pass
    return result
