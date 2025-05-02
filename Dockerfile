FROM nvcr.io/nim/nvidia/nv-embedqa-e5-v5:latest

# Switch to root for installing packages
USER root

RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip


# Then switch back to the original user (if needed)
# USER <your-previous-username-or-UID>

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url=https://pypi.org/simple -r requirements.txt
RUN pip install -U "pydantic>=2.0,<3.0"
RUN pip install "llama-index[faiss]"


COPY . .

EXPOSE 7860
