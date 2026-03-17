# VERI-AI EDGE

## Setup

### Frontend (Vercel)

1. Connect this repo to [Vercel](https://vercel.com)
2. Set the environment variable in the Vercel dashboard:
   ```
   NEXT_PUBLIC_API_URL = https://your-backend-url
   ```
3. Deploy — the frontend builds with `npm run build`.

### Backend (local / Railway / Render)

```bash
# From the repo root
pip install -r requirements.txt
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Required environment variables for the backend:**
- None (model weights are auto-downloaded from Hugging Face on first run)

### Quick local dev

```bash
# Terminal 1 — backend
PYTHONPATH=. uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
npm run dev

# Terminal 3 — tunnel (optional, for mobile testing)
ngrok http 8000
```

Update `.env.local` with your ngrok URL each time it changes:
```
NEXT_PUBLIC_API_URL=https://xxxx.ngrok-free.dev
```
