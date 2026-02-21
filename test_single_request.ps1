# Simpler PowerShell test - just send one request
# Usage: .\test_single_request.ps1

# Replace these with your actual values
$GATEWAY_URL = "https://your-gateway.railway.app"
$GATEWAY_API_KEY = "your-gateway-api-key-here"

# OR set from environment
if ($env:GATEWAY_API_KEY) {
    $GATEWAY_API_KEY = $env:GATEWAY_API_KEY
}

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
    Write-Host "Response:" -ForegroundColor Yellow
    Write-Host $_.Exception.Response -ForegroundColor Gray
}
