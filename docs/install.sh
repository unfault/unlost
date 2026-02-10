#!/bin/bash
# Unlost installer script
# Usage: curl -fsSL https://unlost.unfault.dev/install.sh | bash

set -euo pipefail

REPO="unfault/unlost"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
BINARY_NAME="unlost"

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
        CYGWIN*|MINGW*|MSYS*)
            error "Windows is not supported via this installer."
            error "Please download the binary manually from: https://github.com/${REPO}/releases"
            exit 1
            ;;
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
    
    echo "${arch}-${os}"
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
    
    target=$(detect_target)
    info "Detected target: ${target}"
    
    version=$(get_latest_version)
    info "Latest version: ${version}"
    
    # Construct download URL
    download_url="https://github.com/${REPO}/releases/download/${version}/unlost-${target}"
    
    # Create temporary directory
    tmp_dir=$(mktemp -d)
    trap "rm -rf ${tmp_dir}" EXIT
    
    binary_path="${tmp_dir}/unlost"
    
    info "Downloading unlost ${version} for ${target}..."
    if ! curl -fsSL -o "$binary_path" "$download_url"; then
        error "Failed to download binary from: ${download_url}"
        exit 1
    fi
    
    # Make binary executable
    chmod +x "$binary_path"
    
    # Verify the binary works
    if ! "$binary_path" --version &>/dev/null; then
        error "Downloaded binary is not valid"
        exit 1
    fi
    
    # Check if we need sudo
    local use_sudo=""
    if [ -d "$INSTALL_DIR" ] && [ ! -w "$INSTALL_DIR" ]; then
        use_sudo="sudo"
        warn "Installation directory ${INSTALL_DIR} requires sudo privileges"
    fi
    
    # Install the binary
    info "Installing to ${INSTALL_DIR}/${BINARY_NAME}..."
    if ! $use_sudo mv "$binary_path" "${INSTALL_DIR}/${BINARY_NAME}"; then
        error "Failed to install binary to ${INSTALL_DIR}"
        error "You may need to run with sudo or set a different INSTALL_DIR"
        exit 1
    fi
    
    info "Successfully installed unlost ${version}"
    info "Run 'unlost --help' to get started"
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
