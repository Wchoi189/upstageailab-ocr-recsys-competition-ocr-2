#!/usr/bin/env bash
#
# ETK (Experiment Tracker Kit) Installation Script
#
# Installs ETK CLI tool globally and configures environment.
#
# Usage:
#   bash install-etk.sh
#   bash install-etk.sh --uninstall
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ETK_SOURCE="${SCRIPT_DIR}/etk.py"
INSTALL_DIR="${HOME}/.local/bin"
ETK_SYMLINK="${INSTALL_DIR}/etk"

print_banner() {
    echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  ETK Installation - EDS v1.0 CLI Tool    ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
    echo ""
}

check_requirements() {
    echo -e "${BLUE}🔍 Checking requirements...${NC}"

    # Check Python 3
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ ERROR: Python 3 not found${NC}"
        exit 1
    fi

    python_version=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}   ✅ Python ${python_version}${NC}"

    # Check if running from correct directory
    if [ ! -f "${ETK_SOURCE}" ]; then
        echo -e "${RED}❌ ERROR: etk.py not found at ${ETK_SOURCE}${NC}"
        echo -e "${RED}   Please run this script from experiment-tracker directory${NC}"
        exit 1
    fi

    echo -e "${GREEN}   ✅ ETK source found${NC}"
    echo ""
}

install_etk() {
    echo -e "${BLUE}📦 Installing ETK...${NC}"

    # Create install directory if it doesn't exist
    if [ ! -d "${INSTALL_DIR}" ]; then
        mkdir -p "${INSTALL_DIR}"
        echo -e "${GREEN}   ✅ Created ${INSTALL_DIR}${NC}"
    fi

    # Create symlink
    if [ -L "${ETK_SYMLINK}" ] || [ -f "${ETK_SYMLINK}" ]; then
        echo -e "${YELLOW}   ⚠️  ETK already installed, removing old version${NC}"
        rm -f "${ETK_SYMLINK}"
    fi

    ln -s "${ETK_SOURCE}" "${ETK_SYMLINK}"
    echo -e "${GREEN}   ✅ Created symlink: ${ETK_SYMLINK} → ${ETK_SOURCE}${NC}"

    # Make executable
    chmod +x "${ETK_SOURCE}"
    echo -e "${GREEN}   ✅ Made etk.py executable${NC}"

    echo ""
}

configure_path() {
    echo -e "${BLUE}⚙️  Configuring PATH...${NC}"

    # Check if already in PATH
    if echo "$PATH" | grep -q "${INSTALL_DIR}"; then
        echo -e "${GREEN}   ✅ ${INSTALL_DIR} already in PATH${NC}"
    else
        echo -e "${YELLOW}   ⚠️  ${INSTALL_DIR} not in PATH${NC}"

        # Detect shell
        if [ -n "$BASH_VERSION" ]; then
            shell_rc="${HOME}/.bashrc"
        elif [ -n "$ZSH_VERSION" ]; then
            shell_rc="${HOME}/.zshrc"
        else
            shell_rc="${HOME}/.profile"
        fi

        echo -e "${YELLOW}   Adding to ${shell_rc}${NC}"
        echo "" >> "${shell_rc}"
        echo "# ETK (Experiment Tracker Kit)" >> "${shell_rc}"
        echo "export PATH=\"\${HOME}/.local/bin:\${PATH}\"" >> "${shell_rc}"

        echo -e "${GREEN}   ✅ Updated ${shell_rc}${NC}"
        echo -e "${YELLOW}   ⚠️  Run: source ${shell_rc}${NC}"
    fi

    echo ""
}

verify_installation() {
    echo -e "${BLUE}🧪 Verifying installation...${NC}"

    # Test ETK execution
    if "${ETK_SYMLINK}" --version &> /dev/null; then
        version=$("${ETK_SYMLINK}" --version 2>&1)
        echo -e "${GREEN}   ✅ ETK installed successfully: ${version}${NC}"
    else
        echo -e "${RED}   ❌ ETK installation verification failed${NC}"
        exit 1
    fi

    echo ""
}

print_usage() {
    echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           ETK Quick Start                ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Initialize new experiment:${NC}"
    echo -e "  etk init my_experiment_name"
    echo ""
    echo -e "${BLUE}Create artifacts:${NC}"
    echo -e "  etk create assessment \"Initial baseline evaluation\""
    echo -e "  etk create report \"Performance metrics\""
    echo -e "  etk create guide \"Setup instructions\""
    echo ""
    echo -e "${BLUE}Check status:${NC}"
    echo -e "  etk status"
    echo -e "  etk list"
    echo ""
    echo -e "${BLUE}Validate compliance:${NC}"
    echo -e "  etk validate"
    echo -e "  etk validate --all"
    echo ""
    echo -e "${BLUE}Get help:${NC}"
    echo -e "  etk --help"
    echo -e "  etk create --help"
    echo ""
}

uninstall_etk() {
    echo -e "${BLUE}🗑️  Uninstalling ETK...${NC}"

    if [ -L "${ETK_SYMLINK}" ] || [ -f "${ETK_SYMLINK}" ]; then
        rm -f "${ETK_SYMLINK}"
        echo -e "${GREEN}   ✅ Removed ${ETK_SYMLINK}${NC}"
    else
        echo -e "${YELLOW}   ⚠️  ETK not found at ${ETK_SYMLINK}${NC}"
    fi

    echo -e "${GREEN}✅ ETK uninstalled${NC}"
    echo ""
}

main() {
    print_banner

    # Check for uninstall flag
    if [ "$1" == "--uninstall" ]; then
        uninstall_etk
        exit 0
    fi

    check_requirements
    install_etk
    configure_path
    verify_installation
    print_usage

    echo -e "${GREEN}✅ Installation complete!${NC}"
    echo ""
    echo -e "${YELLOW}💡 Tip: If 'etk' command not found, run: source ~/.bashrc (or ~/.zshrc)${NC}"
    echo ""
}

main "$@"
