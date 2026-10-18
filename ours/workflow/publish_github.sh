#!/usr/bin/env bash
# Publish the source tree while preserving existing remote history.
set -euo pipefail

source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
repository="zjuDBxAI/REMORA"
repository_url="https://github.com/${repository}.git"
existing_checkout=""
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--checkout" ]]; then
    printf 'Usage: bash %s [--checkout /path/to/REMORA]\n' "${BASH_SOURCE[0]}" >&2
    exit 1
  fi
  existing_checkout="$2"
fi

for program in git gh python3; do
  if ! command -v "$program" >/dev/null 2>&1; then
    printf 'Required program is missing: %s\n' "$program" >&2
    exit 1
  fi
done

if ! gh auth status --hostname github.com >/dev/null 2>&1; then
  printf 'Authenticate first: gh auth login --hostname github.com\n' >&2
  exit 1
fi

default_branch="$(gh api "repos/${repository}" --jq .default_branch)"
publish_root="$(mktemp -d "${TMPDIR:-/tmp}/remora-publish.XXXXXX")"
checkout="${publish_root}/checkout"
snapshot_git="${publish_root}/snapshot.git"
printf 'Publishing to %s on branch %s\n' "$repository_url" "$default_branch"

if [[ -n "$existing_checkout" ]]; then
  checkout="$(cd -- "$existing_checkout" && pwd)"
  if [[ "$(git -C "$checkout" rev-parse --show-toplevel)" != "$checkout" ]]; then
    printf 'The checkout path must be the repository root.\n' >&2
    exit 1
  fi
  case "$(git -C "$checkout" config --get remote.origin.url)" in
    "$repository_url"|"https://github.com/${repository}"|"git@github.com:${repository}.git") ;;
    *) printf 'The checkout origin must point to %s\n' "$repository" >&2; exit 1 ;;
  esac
  if [[ -n "$(git -C "$checkout" status --porcelain)" ]]; then
    printf 'The checkout has local changes. Commit or save them before publishing.\n' >&2
    exit 1
  fi
  git -C "$checkout" -c credential.helper= -c 'credential.helper=!gh auth git-credential' \
    fetch origin "$default_branch"
  git -C "$checkout" checkout "$default_branch"
  git -C "$checkout" merge --ff-only "origin/${default_branch}"
else
  # Authentication applies to this invocation only; do not rewrite global config.
  git -c credential.helper= -c 'credential.helper=!gh auth git-credential' \
    clone -- "$repository_url" "$checkout"
  if git -C "$checkout" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "$checkout" checkout "$default_branch"
  else
    git -C "$checkout" symbolic-ref HEAD "refs/heads/${default_branch}"
  fi
fi
printf 'Publication checkout: %s\n' "$checkout"

# Build a source snapshot using .gitignore, without writing to the source .git.
git init --bare --quiet "$snapshot_git"
git -C "$source_root" --git-dir="$snapshot_git" --work-tree="$source_root" add --all -- .
snapshot_tree="$(git --git-dir="$snapshot_git" write-tree)"
git --git-dir="$snapshot_git" archive --format=tar --output="${publish_root}/source.tar" "$snapshot_tree"
python3 - "${publish_root}/source.tar" "$checkout" <<'PY'
from pathlib import Path
import sys
import tarfile

target = Path(sys.argv[2]).resolve()
with tarfile.open(sys.argv[1]) as archive:
    members = archive.getmembers()
    # Refuse path traversal and writes through symlinks from an existing repo.
    for member in members:
        path = target / member.name
        if not path.resolve().is_relative_to(target):
            raise RuntimeError(f'Archive path escapes checkout: {member.name}')
        if path.is_symlink() and not (member.issym() and path.readlink() == Path(member.linkname)):
            raise RuntimeError(f'Existing symlink requires review: {member.name}')
        if member.issym():
            resolved_link = (path.parent / member.linkname).resolve()
            if not resolved_link.is_relative_to(target):
                raise RuntimeError(f'Symlink escapes checkout: {member.name}')
    archive.extractall(target)
PY

if ! git -C "$checkout" config user.name >/dev/null || ! git -C "$checkout" config user.email >/dev/null; then
  github_login="$(gh api user --jq .login)"
  github_id="$(gh api user --jq .id)"
  git -C "$checkout" config user.name "$github_login"
  git -C "$checkout" config user.email "${github_id}+${github_login}@users.noreply.github.com"
fi

git -C "$checkout" add --all
if git -C "$checkout" diff --cached --quiet; then
  printf 'The remote already contains this source snapshot. No changes to push.\n'
  exit 0
fi

git -C "$checkout" diff --cached --stat
git -C "$checkout" commit -m 'Organize REMORA method workflow and English documentation'
git -C "$checkout" -c credential.helper= -c 'credential.helper=!gh auth git-credential' \
  push origin "HEAD:refs/heads/${default_branch}"
printf 'Published commit: '
git -C "$checkout" rev-parse HEAD
printf 'Repository: https://github.com/%s\n' "$repository"
