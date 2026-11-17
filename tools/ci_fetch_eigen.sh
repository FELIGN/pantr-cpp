#!/usr/bin/env bash
# Populate an Eigen checkout at the commit cmake/PantrDependencies.cmake pins, for CI.
#
# Usage: tools/ci_fetch_eigen.sh <dir>
#
# Eigen is hosted only on gitlab.com, and CI runs have failed whole when GitLab refused clones
# under load. CI therefore caches <dir> (keyed on cmake/PantrDependencies.cmake) and points CMake
# at it through PANTR_EIGEN_SOURCE_DIR; this script is the cache-miss path. An existing checkout
# at the pinned commit is kept; anything else is fetched again, with retries, and the commit it
# ends on is checked against the pin.
set -euo pipefail

dir=${1:?usage: tools/ci_fetch_eigen.sh <dir>}
deps="$(dirname "$0")/../cmake/PantrDependencies.cmake"

# The pantr_eigen declaration's GIT_REPOSITORY and GIT_TAG lines.
block=$(sed -n '/FetchContent_Declare(/,/)/p' "$deps" | sed -n '/pantr_eigen/,/)/p')
url=$(sed -n 's/^ *GIT_REPOSITORY  *\([^ ]*\).*/\1/p' <<<"$block")
sha=$(sed -n 's/^ *GIT_TAG  *\([0-9a-f]\{40\}\).*/\1/p' <<<"$block")
if [[ -z $url || -z $sha ]]; then
  echo "ci_fetch_eigen: no GIT_REPOSITORY/GIT_TAG for pantr_eigen in $deps" >&2
  exit 1
fi

if [[ -d $dir/.git ]] && [[ $(git -C "$dir" rev-parse HEAD 2>/dev/null) == "$sha" ]]; then
  echo "ci_fetch_eigen: $dir is already at $sha (cache hit)"
  exit 0
fi

if [[ -e $dir ]]; then rm -r "$dir"; fi
git init --quiet "$dir"
for attempt in 1 2 3 4 5; do
  if git -C "$dir" fetch --quiet --depth 1 "$url" "$sha"; then
    git -C "$dir" checkout --quiet FETCH_HEAD
    if [[ $(git -C "$dir" rev-parse HEAD) != "$sha" ]]; then
      echo "ci_fetch_eigen: fetched $(git -C "$dir" rev-parse HEAD), expected $sha" >&2
      exit 1
    fi
    echo "ci_fetch_eigen: fetched $sha from $url (attempt $attempt)"
    exit 0
  fi
  echo "ci_fetch_eigen: fetch attempt $attempt of 5 failed; retrying in $((attempt * 30)) s" >&2
  sleep $((attempt * 30))
done
echo "ci_fetch_eigen: could not fetch $sha from $url after 5 attempts" >&2
exit 1
