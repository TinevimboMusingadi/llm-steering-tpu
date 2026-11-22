"""headersHelper for the Colab API MCP server: prints fresh auth headers as JSON (Claude Code calls it on connect).
Uses the active gcloud account; the token lasts ~1 hour and is re-minted on every reconnect."""
import json
import os
import shutil
import subprocess

gcloud = shutil.which("gcloud") or shutil.which("gcloud.cmd") or "gcloud"
run = lambda *a: subprocess.run([gcloud, *a], capture_output=True, text=True, check=True, shell=os.name == "nt").stdout.strip()
token = run("auth", "print-access-token")
project = os.environ.get("COLAB_MCP_PROJECT") or run("config", "get-value", "project")
print(json.dumps({"Authorization": f"Bearer {token}", "x-goog-user-project": project}))
