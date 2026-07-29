# Frontend

A deliberately small React + Vite page. One customer picker, five panels, no
router, no component library, no state manager. The point is to show that the
four models are reachable over HTTP, not to demonstrate front-end
architecture.

```bash
cd frontend
npm install
cp .env.example .env.local        # points at http://localhost:8000
npm run dev
```

The API must be running:

```bash
uvicorn rci.api:app --reload      # from the repo root
```

## Deploying to Vercel

Root directory `frontend`, framework preset Vite, build command `npm run build`,
output `dist`. Add one environment variable:

```
VITE_API_URL = https://<your-railway-service>.up.railway.app
```

**Vite inlines `VITE_*` variables into the shipped bundle at build time.** That
prefix is a safety rail: anything named that way is readable by anyone who
opens devtools. A public read-only API URL is fine there. A key never is.

Changing `VITE_API_URL` requires a **redeploy**, not just a settings save,
because the value is baked into the built JavaScript rather than read at
runtime.

## Notes

- The API sets `allow_origins=["*"]`, which is why the browser can call
  Railway from a Vercel origin. Tighten that to the Vercel domain if this ever
  writes anything.
- Each panel fetches independently, so one failing endpoint leaves the other
  four rendered rather than blanking the page.
- Every model's caveat renders next to its number, not in a footnote. An
  uncalibrated probability shown without its warning is how a number ends up
  multiplied by a budget in somebody's spreadsheet.
