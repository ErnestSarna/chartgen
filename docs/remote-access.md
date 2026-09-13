# Charting from anywhere: Cloudflare Tunnel + Access

The web UI (`chartgen-web.bat`) turns the PC with the GPU into a private
charting service for you and people you choose. Nothing is exposed on your
router, your home IP is never published, and a login wall sits in front of
the app before any request reaches your machine. All of it is on
Cloudflare's free tier.

How it behaves once it is up:

- Everyone signs in through Cloudflare Access (email one-time PIN, or a
  Google/GitHub login) and only sees their own queue.
- Finished songs download to the visitor's device automatically.
- **Owners** (the emails you list in `CHARTGEN_WEB_OWNERS`) get their songs
  saved to the PC's Clone Hero library as well, exactly like the desktop app.
- **Guests** (anyone else on the allowlist) get their song charted in a
  scratch folder, handed over as a zip, and deleted from the PC once they
  have downloaded it. Their uploaded audio is deleted as soon as charting
  finishes. Nothing of theirs stays unless they tick "save to this PC's
  library".
- Anything a guest never downloads is swept after 24 hours.

The PC has to be on for any of this to work; if it sleeps, the tunnel drops.

## 1. Install the tunnel (Cloudflare dashboard + one admin command)

1. Cloudflare dashboard → **Zero Trust** → **Networks** → **Tunnels** →
   **Create a tunnel** → *Cloudflared* → name it `chartgen`.
2. Pick **Windows 64-bit**. It shows a one-line install command containing
   your tunnel token. Run it in an **administrator** PowerShell. That installs
   `cloudflared` as a Windows service that starts with the PC.
3. Under **Public Hostname** → **Add a public hostname**:
   subdomain `app`, domain `chartgen.org`, type **HTTP**,
   URL `localhost:8471`. Save. Cloudflare creates the DNS record itself.

## 2. Put a login in front of it

1. **Zero Trust** → **Access** → **Applications** → **Add an application** →
   *Self-hosted*. Name `chartgen`, domain `app.chartgen.org`.
2. Add a policy: name `allowed`, action **Allow**, include **Emails** and list
   every address that may use it (yours first). Save.
3. Under **Settings** → **Authentication**, make sure **One-time PIN** is
   enabled (it is by default). Visitors get a code by email; no accounts.

Anyone not on the list hits Cloudflare's login page and never reaches your
PC. The app trusts the `Cf-Access-Authenticated-User-Email` header Access
adds, which is why `chartgen-web` listens on `127.0.0.1` only: nothing on
your network can reach it without going through Access.

## 3. Tell chartgen who the owners are

Set the environment variable once (System Properties → Environment
Variables, or in an admin PowerShell):

```
[Environment]::SetEnvironmentVariable("CHARTGEN_WEB_OWNERS", "you@example.com", "User")
```

Comma-separate several addresses. Anyone signed in with another address is
a guest.

## 4. Keep the web UI running

Create a scheduled task that runs at logon:

```
schtasks /Create /TN "chartgen web" /SC ONLOGON /RL LIMITED ^
  /TR "\"C:\path\to\chartgen\.venv\Scripts\pythonw.exe\" -m chartgen.web" ^
  /IT
```

(`pythonw.exe` so there is no console window; the working directory must be
the chartgen folder, so put the task's *Start in* field to it if you create
it through the Task Scheduler UI instead.) Or simply run `chartgen-web.bat`
and leave the window open.

## 5. Test it

From a phone on mobile data: open `https://app.chartgen.org`, sign in with
the emailed code, paste a link. About 90 seconds later the zip arrives.

## Stopping

Disable the Access application (or remove an email from the policy) to cut
someone off instantly; delete the tunnel to take the whole thing down.

## What this is not

A private tool for you and people you know, which is what keeps it on the
right side of the copyright line — the same posture as running chartgen
locally, with a longer cable. It is not a public "paste a link" service:
never widen the Access policy to "everyone".
