# Deploy the split Sovereign workspace

The backend Git repository is `sovereign/`. The standalone website is the sibling
`sovereign-frontend/` folder. The frontend has no Git repository yet, so you can
connect it to a new remote when ready. Neither deployment runs optimization jobs:
the installed worker on your computer or GPU server runs the solver and sends
results back over HTTPS.

## 1. Deploy the backend

Import the `sovereign` Git repository into Vercel. Its root `vercel.json` deploys
the FastAPI coordinator at `/api/*`. Alternatively, use the backend-only Dockerfile
on Render. The backend needs persistent storage: set `DATABASE_URL` to a Postgres
connection string on Vercel. On Render, set `SOVEREIGN_DATABASE_URL` for Postgres
or use a persistent disk with `SOVEREIGN_DB` (see `DEPLOYMENT.md`). Vercel returns
503 if no Postgres URL is set.

Generate a workspace secret with
`python -c "import secrets; print(secrets.token_urlsafe(32))"` and set it as
`SOVEREIGN_ADMIN_TOKEN` on the backend. Set
`SOVEREIGN_PUBLIC_ORIGIN=https://YOUR-FRONTEND.vercel.app` on the backend too,
using the exact production frontend origin with no trailing slash. Redeploy after
changing environment variables. Check `https://YOUR-BACKEND.vercel.app/api/health`.

## 2. Deploy the frontend

Deploy `sovereign-frontend/` as a separate Vercel project. You can use the Vercel
CLI now or connect this folder to its own Git repository later. Set
`SOVEREIGN_BACKEND_URL=https://YOUR-BACKEND.vercel.app` in the frontend Vercel
environment and redeploy. Its `vercel.js` sends `/api/*` to that backend; browser
calls and session cookies remain on the frontend origin. Set the backend's
`SOVEREIGN_PUBLIC_ORIGIN` to the frontend's actual production URL. Open
`https://YOUR-FRONTEND.vercel.app/` and sign in with the workspace secret.

For a manual CLI deployment from `sovereign-frontend/`, link that directory to
a new Vercel project, configure the environment variable, and run `vercel --prod`.
No Git repository is required for that path.

## 3. Connect your GPU computer

On the GPU computer, clone the backend repository and install Python 3.10+, CMake,
the NVIDIA driver, CUDA Toolkit, and a compatible C++ compiler. In that repository:

```powershell
python -m pip install .
cmake -S . -B build-gpu -A x64 -DSOVEREIGN_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DSOVEREIGN_BUILD_TESTS=OFF
cmake --build build-gpu --config Release --target sovereign -j 8
.\build-gpu\solver\Release\sovereign.exe capabilities
```

`cuda_available` must be `true`. Linux builds omit `-A x64` and use
`-DCMAKE_BUILD_TYPE=Release`; the executable is `./build-gpu/solver/sovereign`.
In the website's **Machines** view, create a worker key. Start the connector with
the **frontend** URL and the built solver path:

```powershell
sovereign-worker --server "https://YOUR-FRONTEND.vercel.app" --engine ".\build-gpu\solver\Release\sovereign.exe"
```

Paste the one-time worker key when prompted. The worker makes outbound HTTPS
requests and needs no inbound port or locally hosted website. If the console
script is not on PATH, use `python -m worker.runner` with the same arguments.
Submit a constrained LP with **GPU (CUDA)** selected and inspect its actual GPU
operation count in the result.

## Limits

Vercel Functions cap request and response bodies at 4.5 MB. This deployment
limits model text to about 3.5 MB and worker completions to 4 MB. Larger models
and full million-variable results need object storage and paginated retrieval.
The Postgres connection is opened per request; a transaction advisory lock
serializes worker claims. Keep the database backed up. Rotating the admin token
invalidates browser sessions; deleting a machine revokes its worker key.
