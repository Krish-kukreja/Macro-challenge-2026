FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Our DREAMPlace .so files are compiled for Python 3.10 (cpython-310).
# The base image has Python 3.11, so we install Python 3.10 alongside.

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 + system deps (network available at build time)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Install PyTorch (CUDA 12.4) and deps for Python 3.10
RUN python3.10 -m pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python3.10 -m pip install --no-cache-dir \
    numpy scipy tqdm matplotlib absl-py

# Set working directory
WORKDIR /challenge

# Copy the challenge evaluation infrastructure
COPY macro_place/ /challenge/macro_place/
COPY pyproject.toml /challenge/pyproject.toml
COPY requirements.txt /challenge/requirements.txt

# Copy benchmarks (TILOS ICCAD04 testcases)
COPY external/MacroPlacement/Testcases/ICCAD04/ /challenge/external/MacroPlacement/Testcases/ICCAD04/

# Copy PlacementCost evaluator
COPY external/MacroPlacement/CodeElements/Plc_client/ /challenge/external/MacroPlacement/CodeElements/Plc_client/

# Copy dreamplace_integration (diff_proxy_optimizer, abu5_shifter, etc.)
COPY dreamplace_integration/ /challenge/dreamplace_integration/

# Install challenge package with Python 3.10
RUN python3.10 -m pip install --no-cache-dir -e .

# Copy submission (placer.py + dreamplace_bundle/)
COPY submissions/analytical_placer/ /submission/

# Make dreamplace_integration importable from /submission context
ENV PYTHONPATH="/challenge:${PYTHONPATH}"

# Default entrypoint: evaluate the submission placer
ENTRYPOINT ["python3.10", "-m", "macro_place.evaluate", "/submission/placer.py"]
CMD ["--all"]
