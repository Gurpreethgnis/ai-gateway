# Simpler PowerShell test - just send one request
# Usage: .\test_single_request.ps1

# Use environment variables if set; otherwise replace these with your actual values
$GATEWAY_URL = if ($env:GATEWAY_URL) { $env:GATEWAY_URL } else { "https://your-gateway.railway.app" }
$GATEWAY_API_KEY = if ($env:GATEWAY_API_KEY) { $env:GATEWAY_API_KEY } else { "your-gateway-api-key-here" }

$body = @{
    model = "claude-sonnet-4-0"
    messages = @(
        @{
            role = "user"
            content = "Hello! Please respond with a brief greeting."
        }
    )
    max_tokens = 100
} | ConvertTo-Json -Depth 10

$headers = @{
    "Authorization" = "Bearer $GATEWAY_API_KEY"
    "Content-Type" = "application/json"
}
# Send X-Debug: 1 to get the real error message in the response when gateway returns 500
if ($env:GATEWAY_DEBUG -eq "1") {
    $headers["X-Debug"] = "1"
}

Write-Host "Sending request to $GATEWAY_URL..." -ForegroundColor Cyan

try {
    $response = Invoke-RestMethod -Uri "$GATEWAY_URL/v1/chat/completions" -Method Post -Headers $headers -Body $body
    Write-Host "SUCCESS!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Response:" -ForegroundColor Yellow
    Write-Host $response.choices[0].message.content -ForegroundColor White
    Write-Host ""
    Write-Host "Model: $($response.model)" -ForegroundColor Gray
    Write-Host "Usage: $($response.usage | ConvertTo-Json -Compress)" -ForegroundColor Gray
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    $body = $_.ErrorDetails.Message
    if ($body) {
        Write-Host "Response body:" -ForegroundColor Yellow
        Write-Host $body -ForegroundColor Gray
        try {
            $json = $body | ConvertFrom-Json -ErrorAction Stop
            if ($json.error_type) {
                Write-Host ""
                Write-Host "Fix: $($json.error_type) - $($json.error_message)" -ForegroundColor Cyan
            }
        } catch {
            # Body is not JSON (e.g. "error code: 502" from proxy)
        }
    }
    Write-Host ""
    Write-Host "502 = upstream/Anthropic error or timeout. Check Railway logs for the real error." -ForegroundColor Yellow
    Write-Host "To see gateway error in response: `$env:GATEWAY_DEBUG = '1'; .\test_single_request.ps1" -ForegroundColor Gray
}
