"""
Utils for handling git submodules 
e.g., for versioning configuration files.
"""
import subprocess


def has_uncommitted_changes(repo_path: str, relative_file_path: str) -> bool:
    """
    Verify if the given file has pending changes in the
    given git repo.

    Params:
        repo_path: str, points to the root folder of the git repo.
        file_name: str, file name

    Returns:
        true if there are uncommitted changes, or false otherwise.
    """
    git_status_output = (
        subprocess.check_output(
            ["git", "-C", repo_path, "status", "--porcelain", relative_file_path]
        )
        .decode("utf-8")
        .strip()
    )

    if git_status_output == "":
        return False

    file_status_code, *_ = git_status_output.split()

    if file_status_code[0] in ["A", "M", "?"]:
        return True

    return False


def get_commit_hash(repo_path: str) -> str:
    """
    Return the 9-hex-character git commit hash for the git repo.

    Params:
        repo_path: str, path to any folder in the git repo.

    Return:
        nine-character hex hash digest of the commit.
    """
    output = subprocess.check_output(["git", "-C", repo_path, "rev-parse", "HEAD"])
    output_string = output.decode("utf-8")

    return output_string[:9]
