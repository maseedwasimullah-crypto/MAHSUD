# MAHSUD local server + public tunnel launcher (Windows).
# Starts the Streamlit app and a Cloudflare quick tunnel, then records the
# current public URL in data/processed/current_public_url.txt.
# NOTE: quick-tunnel URLs are random and change on each restart. For a
# permanent fixed URL, deploy to Streamlit Community Cloud (see README).

$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$temp = Join-Path $env:LOCALAPPDATA "Temp\opencode"
New-Item -ItemType Directory -Force -Path $temp | Out-Null
$cf = Join-Path $temp "cloudflared.exe"
if (-not (Test-Path $cf)) {
    Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $cf -UseBasicParsing
}

if (-not (Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process python -ArgumentList "-m","streamlit","run","app.py","--server.headless","true","--server.port","8501" -WindowStyle Hidden
    Start-Sleep -Seconds 12
}

$log = Join-Path $temp "cf_tunnel.log"
Remove-Item $log -ErrorAction SilentlyContinue
Start-Process $cf -ArgumentList "tunnel","--no-autoupdate","--url","http://localhost:8501" -RedirectStandardError $log -WindowStyle Hidden

for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    $m = Select-String -Path $log -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($m) {
        $url = $m.Matches[0].Value
        $url | Set-Content (Join-Path $root "data\processed\current_public_url.txt")
        Write-Output $url
        break
    }
}
