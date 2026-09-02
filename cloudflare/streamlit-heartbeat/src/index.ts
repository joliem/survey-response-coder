const APP_URL = "https://survey-response-coder.streamlit.app/";
const MAX_REDIRECTS = 10;

function updateCookies(response: Response, cookies: Map<string, string>): void {
  const setCookie = response.headers.get("set-cookie");
  if (!setCookie) return;

  // Fetch does not maintain a cookie jar. Split combined Set-Cookie values while
  // leaving commas inside attributes such as Expires intact.
  for (const value of setCookie.split(/,(?=\s*[^;,=\s]+=)/)) {
    const pair = value.split(";", 1)[0];
    const separator = pair.indexOf("=");
    if (separator === -1) continue;

    const name = pair.slice(0, separator).trim();
    const cookieValue = pair.slice(separator + 1).trim();
    if (cookieValue) cookies.set(name, cookieValue);
    else cookies.delete(name);
  }
}

async function pingStreamlit(): Promise<void> {
  const cookies = new Map<string, string>();
  let url = APP_URL;

  for (let redirectCount = 0; redirectCount <= MAX_REDIRECTS; redirectCount++) {
    const headers = new Headers({
      "User-Agent": "survey-response-coder-heartbeat/1.0",
    });
    if (cookies.size) {
      headers.set(
        "Cookie",
        [...cookies].map(([name, value]) => `${name}=${value}`).join("; "),
      );
    }

    const response = await fetch(url, {
      method: "GET",
      redirect: "manual",
      headers,
    });
    updateCookies(response, cookies);

    if (response.status >= 300 && response.status < 400) {
      const location = response.headers.get("location");
      if (!location) {
        throw new Error("Streamlit heartbeat received a redirect without a location");
      }
      url = new URL(location, url).href;
      continue;
    }

    if (!response.ok) {
      throw new Error(`Streamlit heartbeat failed with HTTP ${response.status}`);
    }

    // Consume the body so Cloudflare can cleanly reuse the connection.
    await response.arrayBuffer();
    console.log(`Streamlit heartbeat succeeded with HTTP ${response.status}`);
    return;
  }

  throw new Error(`Streamlit heartbeat exceeded ${MAX_REDIRECTS} redirects`);
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
