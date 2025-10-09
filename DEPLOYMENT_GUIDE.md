# 🚀 OccolusAI - Production Deployment Guide

Complete guide to deploy both frontend and backend to production.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Backend Deployment (Render)](#backend-deployment-render)
3. [Frontend Deployment (Vercel)](#frontend-deployment-vercel)
4. [Environment Variables Setup](#environment-variables-setup)
5. [Testing Your Deployment](#testing-your-deployment)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you start, make sure you have:

- ✅ GitHub account
- ✅ [Vercel account](https://vercel.com/signup) (free)
- ✅ [Render account](https://render.com/register) (free) OR [Railway account](https://railway.app/) (free)
- ✅ [Google Gemini API Key](https://makersuite.google.com/app/apikey) (free)
- ✅ Your code pushed to GitHub repository

---

## 🔧 Backend Deployment (Render)

### Option 1: Deploy to Render (Recommended - Easier)

#### Step 1: Push Your Code to GitHub

```bash
cd "c:\Users\imash\Occolus AI"
git add .
git commit -m "Ready for deployment"
git push origin master
```

#### Step 2: Create New Web Service on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:

   **Basic Settings:**
   - **Name:** `occolus-ai-backend`
   - **Region:** Choose closest to your users
   - **Branch:** `master`
   - **Root Directory:** `server`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

#### Step 3: Add Environment Variables

In Render dashboard, go to **Environment** tab and add:

| Key | Value |
|-----|-------|
| `PYTHON_VERSION` | `3.11.0` |
| `GEMINI_API_KEY` | Your Gemini API key from [here](https://makersuite.google.com/app/apikey) |
| `CORS_ORIGINS` | `https://your-frontend-url.vercel.app` (add after deploying frontend) |

#### Step 4: Deploy

1. Click **"Create Web Service"**
2. Wait 5-10 minutes for deployment
3. Copy your backend URL (e.g., `https://occolus-ai-backend.onrender.com`)

---

### Option 2: Deploy to Railway (Alternative)

#### Step 1: Push Code to GitHub (same as above)

#### Step 2: Deploy on Railway

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select your repository
4. Railway will auto-detect Python and deploy

#### Step 3: Configure Environment Variables

Add these in Railway dashboard:

| Key | Value |
|-----|-------|
| `GEMINI_API_KEY` | Your Gemini API key |
| `CORS_ORIGINS` | `https://your-frontend-url.vercel.app` |

#### Step 4: Get Your Backend URL

1. Go to **Settings** tab
2. Click **"Generate Domain"**
3. Copy the URL (e.g., `https://your-project.railway.app`)

---

## 🎨 Frontend Deployment (Vercel)

### Step 1: Prepare Frontend Configuration

Your frontend is already configured with:
- ✅ `vercel.json` - Build configuration
- ✅ `.env.example` - Environment variables template
- ✅ TypeScript errors silenced for fast deployment

### Step 2: Create `.env.local` File

In `client` folder, create `.env.local`:

```env
NEXT_PUBLIC_API_URL=https://your-backend-url.onrender.com
```

Replace `your-backend-url.onrender.com` with your actual backend URL from Step above.

### Step 3: Test Build Locally

```bash
cd "c:\Users\imash\Occolus AI\client"
npm run build
```

If successful, you'll see:
```
✓ Compiled successfully
✓ Generating static pages
```

### Step 4: Deploy to Vercel

#### Method A: Using Vercel Dashboard (Easiest)

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click **"Add New..."** → **"Project"**
3. Import your GitHub repository
4. Configure the project:

   **Framework Preset:** `Next.js`
   **Root Directory:** `client`
   **Build Command:** `npm run build`
   **Output Directory:** `out`

5. Add Environment Variable:
   - **Key:** `NEXT_PUBLIC_API_URL`
   - **Value:** `https://your-backend-url.onrender.com`

6. Click **"Deploy"**
7. Wait 2-3 minutes
8. Copy your frontend URL (e.g., `https://occolus-ai.vercel.app`)

#### Method B: Using Vercel CLI (Advanced)

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy
cd "c:\Users\imash\Occolus AI\client"
vercel --prod
```

### Step 5: Update Backend CORS

Go back to your Render/Railway dashboard and update `CORS_ORIGINS`:

```
CORS_ORIGINS=http://localhost:3000,https://your-frontend-url.vercel.app
```

**Important:** Use your actual Vercel URL!

---

## 🔐 Environment Variables Setup

### Backend Environment Variables

Create `server/.env` file:

```env
GEMINI_API_KEY=your_actual_api_key_here
CORS_ORIGINS=http://localhost:3000,https://your-frontend.vercel.app
```

### Frontend Environment Variables

Create `client/.env.local` file:

```env
NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
```

**For Production (Vercel Dashboard):**
- Add `NEXT_PUBLIC_API_URL` in Vercel project settings
- Redeploy after adding variables

---

## ✅ Testing Your Deployment

### 1. Test Backend API

Open your browser and go to:
```
https://your-backend-url.onrender.com/docs
```

You should see FastAPI documentation page.

Test the health endpoint:
```
https://your-backend-url.onrender.com/health
```

Should return: `{"status": "healthy"}`

### 2. Test Frontend

Open your Vercel URL:
```
https://your-frontend.vercel.app
```

You should see the OccolusAI homepage.

### 3. Test Full Integration

1. Go to your frontend URL
2. Try searching for a protein (e.g., "P53")
3. Click "Discover Candidates"
4. Check if results load properly

**If it works:** 🎉 Congratulations! Your app is live!

**If not:** Check the troubleshooting section below.

---

## 🐛 Troubleshooting

### Common Issues

#### 1. CORS Error in Browser Console

**Error:**
```
Access to fetch at 'https://backend.onrender.com' from origin 'https://frontend.vercel.app' 
has been blocked by CORS policy
```

**Fix:**
- Update `CORS_ORIGINS` in your backend environment variables
- Include your exact Vercel URL (with https://)
- Redeploy backend after changing environment variables

#### 2. Backend Fails to Start

**Error:** `No module named 'rdkit'`

**Fix:**
- Render/Railway should install from `requirements.txt` automatically
- Check build logs in dashboard
- Make sure `requirements.txt` is in the `server` folder

#### 3. Frontend Shows "API Connection Error"

**Possible Causes:**
- Wrong `NEXT_PUBLIC_API_URL` in Vercel
- Backend not deployed yet
- Backend crashed (check Render logs)

**Fix:**
1. Verify backend URL is correct and includes `https://`
2. Check backend logs in Render/Railway dashboard
3. Redeploy frontend after fixing URL

#### 4. Images Not Loading

**Fix:**
- Check browser console for errors
- Verify RDKit is installed on backend
- Check network tab to see if API calls are successful

#### 5. Render Free Tier Sleeps

**Issue:** Free tier backends sleep after 15 minutes of inactivity

**Solutions:**
- Use a cron job service to ping your backend every 10 minutes
- Upgrade to paid tier ($7/month for always-on)
- Accept the 30-second cold start on first request

---

## 🔄 Updating Your Deployment

### Update Backend

```bash
cd "c:\Users\imash\Occolus AI"
git add server/
git commit -m "Update backend"
git push origin master
```

Render/Railway will automatically redeploy.

### Update Frontend

```bash
cd "c:\Users\imash\Occolus AI"
git add client/
git commit -m "Update frontend"
git push origin master
```

Vercel will automatically redeploy.

---

## 💰 Cost Summary

| Service | Free Tier | Limitations |
|---------|-----------|-------------|
| **Vercel** | ✅ Yes | 100GB bandwidth/month |
| **Render** | ✅ Yes | Sleeps after 15 min inactivity, 750 hours/month |
| **Railway** | ✅ $5 credit/month | After credit: $0.000231/GB-hr |
| **Gemini API** | ✅ Yes | 60 requests/minute |

**Total Cost:** $0/month for moderate usage! 🎉

---

## 🎯 Quick Deployment Checklist

- [ ] Push code to GitHub
- [ ] Get Gemini API key
- [ ] Deploy backend to Render/Railway
- [ ] Copy backend URL
- [ ] Deploy frontend to Vercel with backend URL
- [ ] Copy frontend URL
- [ ] Update backend CORS with frontend URL
- [ ] Test the live application
- [ ] Celebrate! 🎉

---

## 📞 Need Help?

If you encounter issues:

1. Check the **Logs** in Render/Vercel dashboard
2. Verify all environment variables are set correctly
3. Make sure GitHub repository has latest code
4. Check browser console for errors (F12)

---

## 🚀 Your URLs

After deployment, save these URLs:

```
Frontend: https://_____________________.vercel.app
Backend:  https://_____________________.onrender.com
API Docs: https://_____________________.onrender.com/docs
```

---

**Now go deploy and share your app with the world! 🌍**
