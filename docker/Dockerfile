# Multi-stage Dockerfile for development and production
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    wget \
    vim \
    git \
    openssh-server \
    build-essential \
    python3-dev \
    libpython3-dev \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libsndfile1-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    locales \
    htop \
    tmux \
    jq \
    bash-completion \
    && rm -rf /var/lib/apt/lists/*

# Set locale to UTF-8 with Korean support
RUN locale-gen en_US.UTF-8 ko_KR.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LC_CTYPE=en_US.UTF-8

# Create vscode user
RUN groupadd -g 1000 vscode && \
    useradd -u 1000 -g vscode -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Set up SSH for vscode user
RUN sudo mkdir -p /run/sshd && \
    sudo mkdir -p /home/vscode/.ssh && \
    sudo chmod 700 /home/vscode/.ssh && \
    sudo chown vscode:vscode /home/vscode/.ssh

# Configure SSH to allow password authentication and key-based authentication
RUN sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sudo sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sudo sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config && \
    sudo sed -i 's/#UsePAM yes/UsePAM yes/' /etc/ssh/sshd_config && \
    sudo sed -i 's/#HostKey /HostKey /' /etc/ssh/sshd_config && \
    echo "StrictModes no" | sudo tee -a /etc/ssh/sshd_config

# Generate SSH host keys
RUN sudo ssh-keygen -A

# Create authorized_keys file
RUN touch /home/vscode/.ssh/authorized_keys && \
    sudo chmod 600 /home/vscode/.ssh/authorized_keys && \
    sudo chown vscode:vscode /home/vscode/.ssh/authorized_keys

# Install uv system-wide
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/opt/uv sh
ENV PATH="/opt/uv/bin:$PATH"

# Set up project directory
RUN mkdir -p /workspaces && chown vscode:vscode /workspaces

USER vscode
WORKDIR /workspaces

# Copy dependency files first for better caching
COPY --chown=vscode:vscode pyproject.toml uv.lock* ./README.md ./

# Note: Dependencies will be installed at runtime to avoid build issues
# RUN uv sync --frozen

# Source code will be mounted via volume at runtime

# Copy .bashrc for enhanced shell experience
COPY --chown=vscode:vscode docker/.bashrc /home/vscode/.bashrc

# Fix Windows line endings in .bashrc
RUN sed -i 's/\r$//' /home/vscode/.bashrc

# Set up development environment
RUN echo 'alias python="uv run python"' >> ~/.bashrc && \
    echo 'alias python3="uv run python3"' >> ~/.bashrc && \
    echo 'alias pytest="uv run pytest"' >> ~/.bashrc && \
    echo 'alias make="uv run make"' >> ~/.bashrc

# Create startup script that starts SSH as root and keeps container running
RUN sudo sh -c 'echo "#!/bin/bash\n\
sudo /usr/sbin/sshd -D & \n\
sleep infinity" > /usr/local/bin/start.sh' && \
    sudo chmod +x /usr/local/bin/start.sh

# Expose ports for development services
EXPOSE 8000 8501 6006 22

# Default command - start SSH daemon
CMD ["/usr/local/bin/start.sh"]
