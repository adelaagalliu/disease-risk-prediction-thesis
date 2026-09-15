# Clarity Health — connected frontend

React + Vite frontend for the FastAPI backend in C:\Users\User\Documents\diploma-uni\backend.

## Run locally

1. Start your existing PostgreSQL service and load your backend environment variables. Docker Compose defaults to host port 5433, while config.py defaults to 5432: DATABASE_URL must match your actual database port. The backend does not automatically read .env.
2. In the backend folder, activate your Python environment, install requirements if necessary, run alembic upgrade head for an uninitialized database, then run uvicorn app.main:app --reload --port 8001.
3. In this frontend folder run pnpm install --frozen-lockfile only if dependencies are missing, then npm.cmd run dev.
4. Open http://localhost:5173 and create a real account or sign in with an existing backend account.

The default VITE_API_URL=/api uses the Vite proxy to http://127.0.0.1:8001. Both dev and preview proxy requests. No CORS change is needed for this setup. Restart the backend to load the new saved-result endpoint.

## What works

- Server registration/login; bearer token stored in sessionStorage for this tab. Passwords and health records are not saved in browser storage by this frontend.
- Empty initial questionnaire requiring explicit answers, exact disease payloads, validation of all sections, and lab units.
- All four predictions are saved by the backend. Successful requests remain saved even if another disease fails; failed cards display the reason. POST requests are never automatically retried.
- Login and Refresh history load database results and full saved explanations. Refresh does not create new assessments.
- History contains up to 500 recent assessments across all diseases; metric charts use up to 250 measurements.
- The profile is reconstructed from saved assessment inputs. Unsaved edits are not persisted. A mixed profile can contain measurements from different dates; review values before recalculating.
- Income labels use the backend's ALL display bands. Result explanations/messages are in the backend's existing Albanian; the surrounding interface remains English.
- Keyboard focus is contained in the edit dialog; Escape closes it when no save is running. Mobile logout remains visible.

## API endpoints

POST /auth/register, POST /auth/login, GET /users/me, POST /predict/{disease}, GET /history/predictions, GET /history/predictions/{id}, GET /history/trends/{metric}.

The saved-result detail endpoint was added to the backend. It enforces account ownership and uses the existing result_payload; no tables or migrations were changed.

## Checks and deployment

Run pnpm test, pnpm build and pnpm preview. If a restricted Windows environment blocks Vite's config bundler, use node node_modules/vite/bin/vite.js build --configLoader native (Node 22.18+).

For a deployed static build, configure the hosting server to proxy /api to FastAPI. Vite's proxy is only a development/preview feature. Alternatively use an HTTPS backend URL in VITE_API_URL and explicitly configure its allowed frontend origin in backend CORS. Do not put database or JWT secrets in VITE_ variables.

Old local prototype accounts/history are not server accounts and are not migrated. Existing localStorage prototype data is left untouched; new code no longer reads it. Register once with the backend if you previously used only the prototype. Backend accounts support email/password, not full names; the dashboard greeting uses the email prefix.
