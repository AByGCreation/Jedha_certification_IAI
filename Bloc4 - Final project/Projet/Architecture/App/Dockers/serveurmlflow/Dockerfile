FROM python:3.10-slim

WORKDIR /home/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends nano unzip curl \
    && rm -rf /var/lib/apt/lists/*

COPY . . 

RUN curl -fsSL https://get.deta.dev/cli.sh | sh

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

# Environment variables will be passed at runtime via docker run -e flags
# No need to set them here in the Dockerfile

#Port 7860 on HFace

CMD mlflow server -p ${PORT} \
    --host 0.0.0.0 \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${ARTIFACT_STORE_URI}