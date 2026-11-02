FROM debian:13-slim

ENV WORKSPACE=/work
ENV TMP_DIR=/temp
ARG TMP_DIR=/temp
ENV PYVERSION=3.12
ENV SSHPORT=2234

# 安装基础依赖、SSH 服务、编译工具链和 LaTeX 工具链
RUN apt update && apt install -y \
    curl unzip git wget openssh-server locales tzdata \
    build-essential procps \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-bibtex-extra \
    biber \
    latexmk \
    ghostscript \
    imagemagick \
    texlive-lang-chinese && \
    chsh -s /bin/bash && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# 配置 SSH 服务（保持原有配置）
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    && sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config \
    && sed -i 's/#Port 22/Port 2234/' /etc/ssh/sshd_config

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装uv python
RUN /bin/bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh && \
    source $HOME/.local/bin/env && \
    uv python install ${PYVERSION} '

# 设置工作目录
WORKDIR $WORKSPACE

# 暴露 SSH 端口
EXPOSE 2234

# 启动命令：bash 执行，激活虚拟环境后启动 SSH 服务
CMD ["/bin/bash", "-c", " /usr/sbin/sshd -D"]
