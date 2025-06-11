s

### 2. Create secrets

kubectl delete secret ngc-secret --namespace=aiea-auditors || true

kubectl create secret docker-registry ngc-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=$NGC_API_KEY \
  --namespace=aiea-auditors

kubectl delete secret api-env-secrets --namespace=aiea-auditors || true

kubectl create secret generic api-env-secrets \
  --from-literal=NGC_API_KEY=$NGC_API_KEY \
  --from-literal=NVIDIA_API_KEY=$NVIDIA_API_KEY \
  --from-literal=OPENAI_API_KEY=$OPENAI_API_KEY \
  --namespace=aiea-auditors
  
### 3. Apply volume and job spec

kubectl apply -f pvc.yaml
kubectl apply -f nvjob.yaml

### 4. Monitor and interact

kubectl get pods -n aiea-auditors
kubectl describe pod embedqa-gpu
kubectl port-forward pod/embedqa-gpu 7862:7862 -n aiea-auditors
kubectl exec -it embedqa-gpu -- /bin/bash


## Files

Ensure the following files are included in your repo before building the Docker image:

app.py

frontend.py

logic.py

vis-transformer.py

best.pt (YOLOv8 weights for vegetable detection)

requirements.txt

With these files copied at build time, no manual kubectl cp steps are required.

Known Fixes

## For compatibility with Gradio v4+ inside the container:

python3 -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/gradio/analytics.py'); text = p.read_text(); text = text.replace('from distutils.version import StrictVersion', 'from packaging.version import Version as StrictVersion'); p.write_text(text)"

## Cleanup

kubectl delete pod embedqa-gpu -n aiea-auditors --grace-period=0 --force