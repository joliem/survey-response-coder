const APP_URL = "https://survey-response-coder.streamlit.app/";

async function pingStreamlit(): Promise<void> {
  const response = await fetch(APP_URL, {
    method: "GET",
    redirect: "follow",
    headers: {
      "User-Agent": "survey-response-coder-heartbeat/1.0",
    },
  });

  if (!response.ok) {
    throw new Error(`Streamlit heartbeat failed with HTTP ${response.status}`);
  }

  // Consume the body so Cloudflare can cleanly reuse the connection.
  await response.arrayBuffer();
  console.log(`Streamlit heartbeat succeeded with HTTP ${response.status}`);
}

export default {
  async scheduled(
    _controller: ScheduledController,
    _env: unknown,
    ctx: ExecutionContext,
  ): Promise<void> {
    ctx.waitUntil(pingStreamlit());
  },

  async fetch(): Promise<Response> {
    await pingStreamlit();
    return new Response("Streamlit heartbeat succeeded\n");
  },
};
