# === STAGE 1: BASE SYSTEM ===
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PNPM_HOME="/usr/local/share/pnpm"

ENV PATH="$PNPM_HOME:$PATH"

# Install ALL system dependencies in one layer to reduce image size and build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo curl wget vim git git-lfs openssh-server build-essential \
    python3-dev libpython3-dev cmake pkg-config \
    libjpeg-dev libpng-dev libsndfile1-dev ffmpeg \
    libsm6 libxext6 libgl1-mesa-glx libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
    locales htop tmux jq lsof bash-completion libcudnn9-cuda-12 tree \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pnpm \
    && rm -rf /var/lib/apt/lists/*

# Setup Locales
RUN locale-gen en_US.UTF-8 ko_KR.UTF-8

# Install GitHub CLI & YQ & UV
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update && apt-get install -y gh \
    && wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/local/bin/yq && chmod +x /usr/local/bin/yq \
    && curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/opt/uv sh

ENV PATH="/opt/uv:$PATH"

# Setup 'vscode' user and SSH config
RUN groupadd -g 1000 vscode && \
    useradd -u 1000 -g vscode -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    mkdir -p /run/sshd && \
    # Configure SSH Daemon
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config && \
    sed -i 's/#StrictModes yes/StrictModes no/' /etc/ssh/sshd_config && \
    # sed -i 's/#LogLevel INFO/LogLevel DEBUG3/' /etc/ssh/sshd_config && \
    sed -i 's/session\s*required\s*pam_loginuid.so/session optional pam_loginuid.so/g' /etc/pam.d/sshd && \
    sed -i 's/#UsePAM yes/UsePAM yes/' /etc/ssh/sshd_config && \
    ssh-keygen -A

# === STAGE 2: DEPENDENCIES ===
FROM base AS dependencies
USER vscode
WORKDIR /workspaces

# Copy project definition files
COPY --chown=vscode:vscode pyproject.toml uv.lock* README.md ./

# Install dependencies into the user's cache
# We use --no-dev because we are in the dependencies stage,
# but you can toggle this if you need dev tools in the base image.
RUN --mount=type=cache,target=/home/vscode/.cache/uv,uid=1000,gid=1000 \
    uv sync --no-dev || uv sync

# === STAGE 3: DEVELOPMENT ===
FROM dependencies AS development

# Switch back to root to install global dev tools
USER root

# Install development tools
RUN pnpm add -g repomix

# Copy Entrypoint Script
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY --chown=vscode:vscode docker/.bashrc /home/vscode/.bashrc

# Fix Windows line endings just in case
RUN sed -i 's/\r$//' /home/vscode/.bashrc && \
    sed -i 's/\r$//' /usr/local/bin/entrypoint.sh

# Switch to user to install pyenv
USER vscode
RUN curl https://pyenv.run | bash

ENV PATH="/home/vscode/.pyenv/bin:$PATH"

# Configure User Environment (Aliases)
RUN echo 'alias python="uv run python"' >> ~/.bashrc && \
    echo 'alias pytest="uv run pytest"' >> ~/.bashrc

# Final Setup
EXPOSE 8000 8501 6006 22
CMD ["/usr/local/bin/entrypoint.sh"]
