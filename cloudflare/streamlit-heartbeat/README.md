# Streamlit heartbeat

This small, separate Cloudflare Worker sends a read-only request to the public
Survey Response Coder app every six hours. Streamlit Community Cloud currently
hibernates apps after 12 hours without traffic, so the schedule leaves a
comfortable margin between requests.

The Worker has no secrets and does not read, create, or change user data. Its
manual HTTP endpoint runs the same check as the scheduled event.

## Deploy or update

From the repository root, while signed in to Cloudflare:

```sh
npx wrangler deploy --config cloudflare/streamlit-heartbeat/wrangler.jsonc
```

## Test immediately

After deploying, open the Worker's `workers.dev` URL. A successful request
returns `Streamlit heartbeat succeeded` and appears in the Cloudflare logs.

Cloudflare cron schedules use UTC. This schedule runs at 00:17, 06:17, 12:17,
and 18:17 UTC each day.
