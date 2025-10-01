### 1. Evironmental Variables
```
export NGC_API_KEY="your-ngc-api-key"
export NVIDIA_API_KEY="your-nvidia-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

### 1.5. Handle Docker Setup (only if first time running on machine)
Build the docker image:
```
docker build --platform=linux/amd64 -t lfaris1234/rootwise-nim-app:latest .
```
Run the container with mounted volume and environment variables:
```
docker run --platform=linux/amd64 \
  -v $(pwd):/app \
  -p 7860:7860 \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NVIDIA_API_KEY=$NVIDIA_API_KEY \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  lfaris1234/rootwise-nim-app:latest
```
Push image to Docker hub:
```
docker push lfaris1234/rootwise-nim-app:latest
```

### 2. Create secrets
```
kubectl delete secret ngc-secret --namespace=aiea-auditors || true

kubectl create secret docker-registry ngc-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=$NGC_API_KEY \
  --namespace=aiea-auditors
```
```
kubectl delete secret api-env-secrets --namespace=aiea-auditors || true

kubectl create secret generic api-env-secrets \
  --from-literal=NGC_API_KEY=$NGC_API_KEY \
  --from-literal=NVIDIA_API_KEY=$NVIDIA_API_KEY \
  --from-literal=OPENAI_API_KEY=$OPENAI_API_KEY \
  --namespace=aiea-auditors
```
### 3. Apply volume and job spec

kubectl apply -f pvc.yaml
kubectl apply -f nvjob.yaml

### 4. Manage and Access Pod
Check pod status:
```
kubectl get pods -n aiea-auditors
kubectl describe pod embedqa-gpu
```
Port forward:
```
kubectl port-forward pod/embedqa-gpu 7862:7862 -n aiea-auditors
```
Access the pod:
```
kubectl exec -it embedqa-gpu -- /bin/bash
```

## Files

Ensure the following files are included in your repo before building the Docker image:

# app.py
Main function, all logic is imported. 

# frontend.py
Gradio frontend features. 

# logic.py
The meat and potatoes, this is where the query engine and completions model at initialized and accessed. 

# vis-transformer.py
Modular implementation of the vision transformer.

# best.pt 
YOLOv8 weights for vegetable detection.

# requirements.txt
A file to track dependencies. 

With these files copied at build time, no manual kubectl cp steps are required, but if there are individual files revised use one or some of the following commands:
```
kubectl cp app.py embedqa-gpu:app.py -n aiea-auditors
kubectl cp frontend.py embedqa-gpu:frontend.py -n aiea-auditors
kubectl cp logic.py embedqa-gpu:logic.py -n aiea-auditors
kubectl cp vis-transformer.py embedqa-gpu:vis-transformer.py -n aiea-auditors
kubectl cp best.pt embedqa-gpu:. -n aiea-auditors
```

## Known fix, Gradio patch: for compatibility with Gradio v4+ inside the container:

python3 -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/gradio/analytics.py'); text = p.read_text(); text = text.replace('from distutils.version import StrictVersion', 'from packaging.version import Version as StrictVersion'); p.write_text(text)"

### 6. Cleanup
```
kubectl delete pod embedqa-gpu -n aiea-auditors --grace-period=0 --force
kubectl delete pvc rootwise-pvc -n aiea-auditors
```