# building on the nim
FROM nvcr.io/nim/nvidia/nv-embedqa-e5-v5:latest

USER root

# is this necessary?
ENV DEBIAN_FRONTEND=noninteractive

# upgrade pip and make necessary installs 
RUN python3 -m pip install --index-url=https://pypi.org/simple --upgrade pip
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --index-url=https://pypi.org/simple -r requirements.txt\
    && pip install -U "pydantic>=2.0,<3.0" "llama-index[faiss]"

# copy entire directory into image
WORKDIR /app
COPY . .

EXPOSE 7860