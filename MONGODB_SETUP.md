# MongoDB setup for the Render mediator

The backend stores jobs, machines, and results in MongoDB. It does **not** run the solver.

## Render environment variables

Set these on `https://sovereign-we6b.onrender.com`:

| Variable | Value |
|---|---|
| `SOVEREIGN_ADMIN_TOKEN` | your workspace sign-in key (24+ chars) |
| `SOVEREIGN_PUBLIC_ORIGIN` | `http://127.0.0.1:5173` for local preview, or your Vercel URL when deployed |
| `MONGODB_URI` | your Atlas connection string (see below) |
| `SOVEREIGN_MONGODB_DB` | `sovereign` (optional; this is the default) |

## Your MongoDB connection string

Replace `<db_password>` with your real Atlas password, and add the database name `sovereign` before the `?`:

```
mongodb://garvit:YOUR_REAL_PASSWORD@learning-shard-00-00.3u2np.mongodb.net:27017,learning-shard-00-01.3u2np.mongodb.net:27017,learning-shard-00-02.3u2np.mongodb.net:27017/sovereign?ssl=true&replicaSet=atlas-ne7nol-shard-0&authSource=admin&appName=Learning
```

Paste that full string as `MONGODB_URI` in Render → Environment → Save → Redeploy.

**Security:** Do not commit this string to Git. If the password was shared in chat, rotate it in MongoDB Atlas.

## Verify

After redeploy, open:

```
https://sovereign-we6b.onrender.com/api/health
```

Expected:

```json
{"status":"ok","mode":"mediator","storage":"mongodb","compute":"remote-workers-only"}
```

If `storage` is still `sqlite`, `MONGODB_URI` is missing or the service was not redeployed.

## Architecture

`Website -> Render mediator (MongoDB) -> sovereign CLI on your PC -> solver -> mediator -> website`

All CPU/GPU computation stays on the machine running `sovereign connect`.
