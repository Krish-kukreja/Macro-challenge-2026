FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Our DREAMPlace .so files are compiled for Python 3.10 (cpython-310).
# Use the PyTorch image which has Python 3.11, but install Python 3.10
# alongside and use it for our submission.

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Install PyTorch and deps for Python 3.10
RUN python3.10 -m pip install --no-cache-dir \
    torch==2.5.1 \
    numpy \
    scipy \
    tqdm \
    matplotlib \
    absl-py

# Set working directory
WORKDIR /challenge

# Copy the challenge evaluation infrastructure
COPY macro_place/ /challenge/macro_place/
COPY pyproject.toml /challenge/pyproject.toml
COPY requirements.txt /challenge/requirements.txt

# Copy benchmarks
COPY external/MacroPlacement/Testcases/ICCAD04/ /challenge/external/MacroPlacement/Testcases/ICCAD04/

# Copy PlacementCost evaluator
COPY external/MacroPlacement/CodeElements/Plc_client/ /challenge/external/MacroPlacement/CodeElements/Plc_client/

# Install challenge package
RUN python3.10 -m pip install --no-cache-dir -e .

# Copy submission
COPY submissions/analytical_placer/ /submission/

# Entrypoint uses Python 3.10
ENTRYPOINT ["python3.10", "-m", "macro_place.evaluate"]
