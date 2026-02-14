#!/bin/bash
# Unlost installer script
# Usage: curl -fsSL https://unlost.unfault.dev/install.sh | bash

set -euo pipefail

REPO="unfault/unlost"
BINARY_BASENAME="unlost"
INSTALL_DIR="${INSTALL_DIR:-}"
EXE_EXT=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print functions
info() {
    echo -e "${GREEN}==>${NC} $1"
}

warn() {
    echo -e "${YELLOW}==>${NC} $1"
}

error() {
    echo -e "${RED}==>${NC} $1" >&2
}

# Detect OS and architecture
detect_target() {
    local os
    local arch
    
    # Detect OS
    case "$(uname -s)" in
        Linux*)     os="unknown-linux-gnu" ;;
        Darwin*)    os="apple-darwin" ;;
        CYGWIN*|MINGW*|MSYS*) os="pc-windows-msvc" ;;
        *)
            error "Unsupported operating system: $(uname -s)"
            exit 1
            ;;
    esac
    
    # Detect architecture
    case "$(uname -m)" in
        x86_64)     arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        *)
            error "Unsupported architecture: $(uname -m)"
            exit 1
            ;;
    esac
    
    # macOS only has aarch64 builds currently
    if [ "$os" = "apple-darwin" ] && [ "$arch" = "x86_64" ]; then
        warn "x86_64 macOS is not officially supported."
        warn "You may need to build from source: cargo install unlost"
        exit 1
    fi

    if [ "$os" = "pc-windows-msvc" ]; then
        EXE_EXT=".exe"
    fi
    
    echo "${arch}-${os}"
}

default_install_dir() {
    # Prefer a user-local bin directory.
    # - Linux/macOS: ~/.local/bin
    # - Windows (Git Bash/MSYS/Cygwin): ~/.local/bin (maps to %USERPROFILE%\\.local\\bin)
    case "$(uname -s)" in
        CYGWIN*|MINGW*|MSYS*)
            echo "${HOME}/.local/bin"
            ;;
        *)
            echo "${HOME}/.local/bin"
            ;;
    esac
}

ensure_install_dir() {
    local dir="$1"
    if [ -z "$dir" ]; then
        error "INSTALL_DIR resolved to an empty path"
        exit 1
    fi

    if [ ! -d "$dir" ]; then
        mkdir -p "$dir" 2>/dev/null || {
            if command -v sudo &>/dev/null; then
                sudo mkdir -p "$dir" 2>/dev/null || true
            fi
        }
    fi

    if [ ! -d "$dir" ]; then
        error "Failed to create install directory: ${dir}"
        error "Set INSTALL_DIR to a writable location and re-run"
        exit 1
    fi
}

path_hint() {
    local dir="$1"
    if command -v python3 &>/dev/null; then
        python3 - "$dir" <<'PY'
import os, sys
dir = sys.argv[1]
paths = os.environ.get('PATH','').split(os.pathsep)
print('yes' if dir in paths else 'no')
PY
        return
    fi

    case ":$PATH:" in
        *":${dir}:"*) echo "yes" ;;
        *) echo "no" ;;
    esac
}

# Get the latest release version
get_latest_version() {
    local version
    version=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null | \
        grep '"tag_name":' | \
        sed -E 's/.*"([^"]+)".*/\1/')
    
    if [ -z "$version" ]; then
        error "Failed to get latest version from GitHub API"
        exit 1
    fi
    
    echo "$version"
}

# Download and install the binary
install_unlost() {
    local target
    local version
    local download_url
    local tmp_dir
    local binary_path
    local install_name
    local ps_install_dir
    
    target=$(detect_target)
    info "Detected target: ${target}"
    
    version=$(get_latest_version)
    info "Latest version: ${version}"
    
    # Construct download URL
    download_url="https://github.com/${REPO}/releases/download/${version}/unlost-${target}${EXE_EXT}"
    
    # Create temporary directory
    tmp_dir=$(mktemp -d)
    trap "rm -rf ${tmp_dir}" EXIT
    
    binary_path="${tmp_dir}/${BINARY_BASENAME}${EXE_EXT}"
    
    info "Downloading unlost ${version} for ${target}..."
    if ! curl -fsSL -o "$binary_path" "$download_url"; then
        error "Failed to download binary from: ${download_url}"
        exit 1
    fi
    
    # Make binary executable
    chmod +x "$binary_path" 2>/dev/null || true
    
    # Verify the binary works
    if ! "$binary_path" --version &>/dev/null; then
        error "Downloaded binary is not valid"
        exit 1
    fi

    if [ -z "${INSTALL_DIR}" ]; then
        INSTALL_DIR="$(default_install_dir)"
    fi

    ensure_install_dir "${INSTALL_DIR}"
    install_name="${BINARY_BASENAME}${EXE_EXT}"
    
    # Check if we need sudo
    local use_sudo=""
    if [ -d "$INSTALL_DIR" ] && [ ! -w "$INSTALL_DIR" ]; then
        use_sudo="sudo"
        warn "Installation directory ${INSTALL_DIR} requires sudo privileges"
    fi
    
    # Install the binary
    info "Installing to ${INSTALL_DIR}/${install_name}..."
    if ! $use_sudo mv "$binary_path" "${INSTALL_DIR}/${install_name}"; then
        error "Failed to install binary to ${INSTALL_DIR}"
        error "You may need to run with sudo or set a different INSTALL_DIR"
        exit 1
    fi
    
    info "Successfully installed unlost ${version}"
    info "Run 'unlost --help' to get started"

    if [ "$(path_hint "${INSTALL_DIR}")" = "no" ]; then
        warn "${INSTALL_DIR} is not on your PATH"
        warn "Add it to PATH (bash/zsh): export PATH=\"${INSTALL_DIR}:\$PATH\""

        ps_install_dir="${INSTALL_DIR}"
        if [ -n "${EXE_EXT}" ] && command -v cygpath &>/dev/null; then
            ps_install_dir="$(cygpath -w "${INSTALL_DIR}")"
        fi

        warn "Add it to PATH (PowerShell): [Environment]::SetEnvironmentVariable('Path', \$env:Path + ';${ps_install_dir}', 'User')"
    fi
}

# Main
main() {
    info "Unlost Installer"
    info "Repository: https://github.com/${REPO}"
    echo ""
    
    # Check for required commands
    if ! command -v curl &>/dev/null; then
        error "curl is required but not installed"
        exit 1
    fi
    
    # Check if unlost is already installed
    if command -v unlost &>/dev/null; then
        local current_version
        current_version=$(unlost --version 2>/dev/null || echo "unknown")
        warn "unlost is already installed: ${current_version}"
        warn "This installer will overwrite the existing installation"
        echo ""
    fi
    
    install_unlost
}

main "$@"
