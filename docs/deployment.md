# Deployment Guide

## Local (recommended for development)

```bash
# 1. Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000

# 2. Seed demo data (new terminal)
python -m scripts.seed_data

# 3. Frontend (new terminal)
cd frontend && npm install && npm run dev
```

---

## Docker Compose

```bash
cp .env.example .env   # fill in OPENAI_API_KEY
docker compose up --build
```

Then open http://localhost:5173

---

## Railway (backend)

1. Create new Railway project → Deploy from GitHub
2. Set root directory: `/` (uses Dockerfile)
3. Add env vars from `.env.example`
4. Railway auto-assigns a URL — set `CORS_ORIGINS` to include your frontend URL

---

## Render

**Backend:**
- New Web Service → connect repo
- Environment: Docker
- Set all env vars

**Frontend:**
- New Static Site → connect repo
- Build command: `cd frontend && npm install && npm run build`
- Publish directory: `frontend/dist`
- Add environment variable: `VITE_API_BASE=<your render backend URL>`

---

## Vercel (frontend only)

```bash
cd frontend
npx vercel --prod
```

Set `VITE_API_BASE` in Vercel dashboard to point to your deployed backend.
