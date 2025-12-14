# Demo Deployment - Quick Start

## For Kaggle Vibe Code with Gemini 3 Competition

### What You Need (Minimum Bundle)

**Total Size**: ~4MB source code + ~50MB with dependencies

**Files Required**:
1. ✅ `frontend/` - React app (already exists)
2. ✅ `backend/` - FastAPI server (already exists)
3. ✅ `demo_data/` - 5 sample artifacts (CREATED)
4. ✅ `demo_scripts/` - 3 validation stubs (CREATED)
5. ✅ `DEMO_DEPLOYMENT_GUIDE.md` - Full deployment guide (CREATED)
6. ✅ `create_demo_data_simple.sh` - Setup script (CREATED)

---

## Quick Deploy (3 Commands)

```bash
# 1. Create demo data
./create_demo_data_simple.sh

# 2. Set demo mode
export DEMO_MODE=true

# 3. Start servers
make dev
```

**Access**: http://localhost:3000

---

## What Works in Demo Mode

### ✅ Fully Functional Features:
1. **Artifact Generator** (Page 1)
   - AI-powered artifact creation using Gemini API
   - Types: Plans, Assessments, Bug Reports, Audits, Designs
   - Real-time frontmatter generation

2. **Framework Auditor** (Page 2)
   - Run validation on demo artifacts
   - Shows compliance status
   - Displays validation reports from stubs

3. **Integration Hub** (Page 3)
   - Displays tracking database status (from stub)
   - Real-time metrics visualization
   - Health check indicators

4. **Strategy Dashboard** (Page 4)
   - Framework health metrics
   - Architecture recommendations
   - Quality scores

5. **Context Explorer** (Page 5)
   - Visualize 5 demo artifacts
   - Show relationships and dependencies
   - Interactive artifact browser

6. **Librarian** (Page 6)
   - Browse demo artifacts by type
   - Search and filter
   - Preview artifact content

7. **Settings** (Page 7)
   - Configure Gemini API key
   - Theme selection
   - Preferences management

### 📋 Demo Artifacts Included:

1. **Implementation Plan**: OCR feature enhancement (active)
2. **Assessment**: Model performance analysis (complete)
3. **Bug Report**: Unicode encoding error (resolved)
4. **Audit**: Code quality review (active)
5. **Design**: API architecture (active)

Each artifact has proper frontmatter and follows AgentQMS conventions.

---

## Deployment Options

### Option 1: Google Cloud Run (Free Tier - RECOMMENDED)

```bash
# Prerequisites: gcloud CLI installed
gcloud auth login

# Deploy (one command!)
gcloud run deploy agentqms-dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DEMO_MODE=true,GEMINI_API_KEY=your_key_here

# Get URL
gcloud run services describe agentqms-dashboard --format='value(status.url)'
```

**Free Tier**: 2M requests/month, 360K GB-seconds/month

### Option 2: Vercel (Frontend) + Railway (Backend)

**Frontend (Vercel)**:
```bash
cd frontend
npm install -g vercel
vercel --prod
# Set VITE_API_URL to Railway backend URL
```

**Backend (Railway)**:
```bash
# Install Railway CLI
npm i -g @railway/cli

# Deploy
railway login
railway init
railway up

# Set environment
railway variables set DEMO_MODE=true
```

### Option 3: Render (All-in-One - Easiest)

Create `render.yaml`:
```yaml
services:
  - type: web
    name: agentqms-dashboard
    env: docker
    plan: free
    envVars:
      - key: DEMO_MODE
        value: true
      - key: GEMINI_API_KEY
        sync: false  # Set in Render dashboard
```

Push to GitHub, connect to Render, deploy automatically.

### Option 4: Hugging Face Spaces (AI Community)

```bash
# Create Space on HF
# Set to "Docker" runtime

# Add Dockerfile (already included)
git push hf main
```

Perfect for Gemini API showcase!

---

## Environment Variables

### Required:
- `DEMO_MODE=true` - Use demo artifacts and stubs
- `GEMINI_API_KEY=xxx` - Your Gemini API key

### Optional:
- `PORT=8000` - Backend port (default: 8000)
- `FRONTEND_URL=http://localhost:3000` - CORS origin

---

## Docker Deployment

```bash
# Build image
docker build -t agentqms-dashboard .

# Run locally
docker run -p 8000:8000 \
  -e DEMO_MODE=true \
  -e GEMINI_API_KEY=your_key \
  agentqms-dashboard

# Access
open http://localhost:8000
```

---

## What's Different in Demo Mode

| Feature | Production | Demo Mode |
|---------|-----------|-----------|
| Artifacts | Read from `docs/artifacts/` | Read from `demo_data/artifacts/` |
| Validation | Runs real AgentQMS tools | Uses `validate_stub.py` |
| Tracking DB | Queries real SQLite DB | Uses `tracking_stub.py` |
| File System | Full access to workspace | Limited to demo_data/ |

**Why Demo Mode?**
- No dependencies on AgentQMS framework installation
- No need for tracking database setup
- Portable - works anywhere
- Perfect for competition judges to test quickly

---

## Competition Submission Checklist

### For Kaggle Vibe Code with Gemini 3:

- [ ] **Live Demo URL** deployed (use one of the options above)
- [ ] **GitHub README** updated with:
  - [ ] Live demo link
  - [ ] Screenshots of each page
  - [ ] Architecture diagram
  - [ ] "How Gemini is Used" section
- [ ] **Video Demo** (2-3 minutes):
  - [ ] Show Artifact Generator creating plan with Gemini
  - [ ] Run validation on generated artifact
  - [ ] Display tracking status
  - [ ] Navigate between pages
- [ ] **Code Highlights**:
  - [ ] `frontend/services/aiService.ts` - Gemini integration
  - [ ] `frontend/components/ArtifactGenerator.tsx` - AI-powered UI
  - [ ] `backend/routes/tools.py` - Tool execution
- [ ] **Documentation**:
  - [ ] DEMO_DEPLOYMENT_GUIDE.md (comprehensive)
  - [ ] README.md (quick start)
  - [ ] API documentation (OpenAPI at /docs)

---

## Tips for Judges

**Testing the Demo** (3-minute walkthrough):

1. **Visit live demo URL** → See dashboard home
2. **Click "Artifact Generator"** → Create implementation plan
   - Enter: Title "Test Feature"
   - Click "Generate with AI"
   - Watch Gemini create structured artifact
3. **Click "Framework Auditor"** → Run validation
   - Click "Quick Validation"
   - See compliance report
4. **Click "Integration Hub"** → View tracking status
   - See active plans, experiments
   - Check system health
5. **Click "Context Explorer"** → Browse artifacts
   - See 5 demo artifacts
   - View frontmatter and content

**What Makes This Special**:
- ✨ **Gemini Integration**: AI generates proper frontmatter, structured content
- 🎯 **Quality Focus**: Built-in validation and compliance checking
- 📊 **Metrics Driven**: Real-time tracking and health monitoring
- 🎨 **Modern Stack**: React 19 + TypeScript + FastAPI

---

## Troubleshooting

### Demo not loading?
- Check `DEMO_MODE=true` is set
- Verify demo data exists: `ls demo_data/artifacts/*/`
- Check backend logs for errors

### Gemini API errors?
- Verify `GEMINI_API_KEY` is set correctly
- Check API quota/billing in Google AI Studio
- Test key with: `curl https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=YOUR_KEY`

### CORS errors?
- Ensure `FRONTEND_URL` matches your frontend origin
- Check CORS middleware in `backend/server.py`

---

## Need Help?

- **Full Guide**: See `DEMO_DEPLOYMENT_GUIDE.md`
- **API Docs**: Visit `/docs` endpoint on backend
- **Source Code**: Everything is in this repository

---

**Estimated Setup Time**: 5 minutes
**Demo Artifact Count**: 5 (expandable)
**API Calls per Demo**: ~10 (lightweight)
**Perfect for**: Competition judges, quick demos, portfolio showcase

🚀 **Ready to deploy!**
