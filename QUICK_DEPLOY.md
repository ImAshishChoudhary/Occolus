# 🚀 Quick Deployment Guide

## ⚡ Super Fast Deployment (15 minutes)

### 1️⃣ Get Gemini API Key (2 min)
1. Go to https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key

### 2️⃣ Deploy Backend to Render (5 min)
1. Go to https://dashboard.render.com/
2. Click "New +" → "Web Service"
3. Connect GitHub repo
4. Settings:
   - **Root Directory:** `server`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variables:
   - `GEMINI_API_KEY` = your API key
   - `PYTHON_VERSION` = `3.11.0`
6. Click "Create Web Service"
7. **Copy your backend URL** (e.g., https://occolus-ai-xyz.onrender.com)

### 3️⃣ Deploy Frontend to Vercel (5 min)
1. Go to https://vercel.com/dashboard
2. Click "Add New..." → "Project"
3. Import GitHub repo
4. Settings:
   - **Root Directory:** `client`
   - **Framework:** Next.js
5. Add Environment Variable:
   - `NEXT_PUBLIC_API_URL` = your backend URL from step 2
6. Click "Deploy"
7. **Copy your frontend URL** (e.g., https://occolus-ai.vercel.app)

### 4️⃣ Update Backend CORS (2 min)
1. Go back to Render dashboard
2. Click on your service → "Environment"
3. Add/Update:
   - `CORS_ORIGINS` = `https://your-frontend-url.vercel.app`
4. Save (auto-redeploys)

### 5️⃣ Test (1 min)
1. Open your Vercel URL
2. Search for "P53"
3. Click "Discover Candidates"
4. ✅ Works? You're live!

---

## 📝 Save Your URLs

```
Frontend: https://_____________________.vercel.app
Backend:  https://_____________________.onrender.com
```

---

## ⚠️ Important Notes

- **First Load:** Backend may take 30 seconds (free tier wakes from sleep)
- **Cost:** $0/month with free tiers
- **Updates:** Just push to GitHub - auto-deploys!

---

## 🆘 Quick Fixes

**CORS Error?**
→ Update `CORS_ORIGINS` in Render with your exact Vercel URL

**API Not Working?**
→ Check `NEXT_PUBLIC_API_URL` in Vercel environment variables

**Backend Sleeping?**
→ First request takes 30s (normal on free tier)

---

**Need detailed help?** → See `DEPLOYMENT_GUIDE.md`
