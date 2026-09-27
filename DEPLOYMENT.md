# Deploy Sovereign on Render and connect your own compute

The backend job queue runs on Render; the separate `sovereign-frontend/` website runs on Vercel. A Python worker on each laptop or company server connects outward over HTTPS. The C++ solver runs on that machine. Results, verification reports, and actual GPU usage return to the website. No inbound firewall port or GPU on Render is required.

## 1. Deploy the backend and website

1. Push this backend repository to GitHub, including `api/cloud.py`, `worker/`, `Dockerfile`, and `render.yaml`.
2. In Render choose **New > Blueprint**, connect the repository, and select `render.yaml`.
3. Review the plan: this configuration uses a **paid Starter web service and a 1 GB persistent disk**. The disk preserves jobs and machine keys across restarts. No service has been provisioned by this change.
4. Deploy. Render runs `api.cloud:app` in Python. Health path: `/api/health`.
5. Deploy the sibling `sovereign-frontend/` folder as a separate Vercel project. Set `SOVEREIGN_BACKEND_URL=https://YOUR-SERVICE.onrender.com` on the frontend.
6. Set `SOVEREIGN_PUBLIC_ORIGIN=https://YOUR-FRONTEND.vercel.app` on Render, using the frontend's exact origin, and redeploy. Open the frontend URL. Copy Render's generated `SOVEREIGN_ADMIN_TOKEN` to sign in.

Keep one service instance. SQLite lives at `/var/data/workspace.db`; horizontal replicas need a shared database redesign. Keep backups of the persistent disk. Free ephemeral storage is suitable only for a disposable preview: jobs and worker registrations will be lost on redeploy.

The frontend forwards `/api/*` to Render, keeping browser session cookies on the website origin. Render terminates HTTPS; the container trusts proxy headers because Render is its ingress. Do not expose the container directly to an untrusted network with that proxy setting.

This is a private, single-workspace product. The workspace key controls all jobs and machines. It is not a multi-tenant billing or user-account system. Worker keys can only claim jobs and report results, not administer the workspace. Treat a connected machine as trusted: it receives model content and runs the solver locally. Rotate the workspace key in Render to invalidate browser sessions. Disconnect a machine in the dashboard to revoke its worker key.

## 2. Build the engine on the compute machine

Install Python 3.10+, CMake 3.24+, Git, and a C++17 compiler. Windows: Visual Studio 2022 with Desktop development with C++. Linux: GCC or Clang. Clone this repository on the compute machine.

### NVIDIA GPU on Windows

Install a current NVIDIA driver and a CUDA Toolkit compatible with your GPU and compiler. The RTX 5050 requires a recent toolkit supporting Blackwell (CUDA 12.8 or newer); use the toolkit's supported compiler version. A driver alone does not provide `nvcc`.

From a Developer PowerShell for Visual Studio, in the repository root:

```powershell
cmake -S . -B build-gpu -A x64 -DSOVEREIGN_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DSOVEREIGN_BUILD_TESTS=OFF
cmake --build build-gpu --config Release --target sovereign -j 8
.\build-gpu\solver\Release\sovereign.exe capabilities
```

`cuda_available` must be `true`. If compiler detection fails, check that `nvcc --version` works and use a toolkit-supported Visual Studio compiler. GPU jobs stay queued when only CPU workers are connected.

### NVIDIA GPU on Linux

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DSOVEREIGN_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DSOVEREIGN_BUILD_TESTS=OFF
cmake --build build-gpu --target sovereign -j 8
./build-gpu/solver/sovereign capabilities
```

### CPU machine (also works without CUDA)

```powershell
cmake -S . -B build-cloud -A x64 -DSOVEREIGN_USE_CUDA=OFF
cmake --build build-cloud --config Release --target sovereign -j 8
.\build-cloud\solver\Release\sovereign.exe capabilities
```

On Linux/macOS omit `-A x64`, add `-DCMAKE_BUILD_TYPE=Release`, and use `./build-cloud/solver/sovereign`. Apple's GPUs are not CUDA devices; macOS workers currently run on CPU.

## 3. Connect a worker

1. In the website, choose **Connect machine**, name it, and create its worker key.
2. Download the worker from that dialog onto the compute machine. It uses Python's standard library; no pip installation is needed on workers.
3. Run the command shown in the dialog, adjusting the engine path. For example:

```powershell
python sovereign-worker.py --server "https://YOUR-FRONTEND.vercel.app" --engine ".\build-gpu\solver\Release\sovereign.exe"
```

Or from the repository:

```powershell
python -m worker.runner --server "https://YOUR-FRONTEND.vercel.app" --engine ".\build-cloud\solver\Release\sovereign.exe"
```

4. Paste the worker key at the hidden prompt. It is not passed as a command-line argument. For a managed service, provide `SOVEREIGN_WORKER_TOKEN`, `SOVEREIGN_SERVER_URL`, and `SOVEREIGN_BIN` in that service's private environment instead.
5. Leave the worker running. It appears online in the dashboard. Use a service manager for a company server; laptop sleep or closing the process makes it offline.

Outbound HTTPS to your frontend hostname is the only network access the worker needs. HTTP is accepted only for loopback development. TLS certificates are verified and credentials are not forwarded to redirects.

## 4. Submit a problem and receive results

Choose **New job**. Upload a JSON/MPS model or select **Use example**. Choose any available machine or a specific worker, select Automatic/CPU/GPU, and submit. The example has optimum **9**.

The API immediately queues the job. The worker claims it, writes the model to a temporary file, starts the C++ CLI, runs independent solution verification, and returns the result. Close the browser if needed; the result is stored on Render. Open the job to inspect variables and download the full result JSON.

- One job runs per worker process. Separate workers on different machines can run jobs concurrently.
- CPU jobs do not invoke CUDA. GPU jobs require a CUDA-ready worker. Automatic jobs may run on CPU or GPU depending on the worker and workload size.
- CUDA accelerates sparse matrix-vector products used by interior-point solver steps. It is **not a full GPU implementation of every LP/MILP/QP algorithm**, and machines are not combined to solve a single model.
- A CUDA-ready machine does not imply a job used GPU kernels. Presolve and some algorithms can finish on CPU. `gpu_used` and `gpu_operations` report actual kernel work; the dashboard does not invent GPU use or speedups.
- The wall-clock limit is enforced by terminating the solver subprocess. Cancellation is picked up at the next heartbeat (normally within 10 seconds). Cancellation or timeout discards partial output.
- A lost worker lease is requeued after 60 seconds, up to three attempts. A stale worker cannot overwrite a new result. A lost response to a result upload is safe to retry.
- Disconnecting a worker revokes its credential. Only queue submissions from authorized workspace users are accepted. Jobs contain model data and fixed solver options, never arbitrary shell commands.
- The dashboard lists the most recent 100 jobs; its summary reflects those jobs. Full older results remain addressable by their job ID. No automated retention policy is configured.

## Local preview and checks

The local preview uses the same coordinator and outbound worker protocol as Render; local operation is only for development.

```powershell
python -m pip install -r requirements.txt
$env:SOVEREIGN_ADMIN_TOKEN = (python -c "import secrets; print(secrets.token_urlsafe(32))")
$env:SOVEREIGN_PUBLIC_ORIGIN = 'http://localhost:5173'
# Copy this key to sign in. Do not commit it.
$env:SOVEREIGN_ADMIN_TOKEN
python -m uvicorn api.cloud:app --host 127.0.0.1 --port 8000
```

In a second terminal run `npm ci` and `npm run dev` from the sibling `sovereign-frontend/` folder, then open `http://localhost:5173/`. Local backend database: `data/workspace.db` (ignored by Git). Vite proxies `/api` to port 8000. Use the website URL when starting a worker.

```powershell
python -m unittest discover -s tests/scripts -p test_cloud.py -v
python tests/scripts/smoke_cloud.py --engine build-cloud/solver/Release/sovereign.exe
ctest --test-dir build-cloud -C Release --output-on-failure
```

The smoke check exercises actual HTTP, the worker process, C++ solves of LP/MILP/QP/MPS models, verification, and result persistence. The CPU benchmark run is recorded in `benchmarks/reports/sih-online.md`, including failures and timeouts. It does not establish a GPU speedup.
