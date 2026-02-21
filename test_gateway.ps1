# PowerShell script to test AI Gateway cost savings
# Usage: .\test_gateway.ps1

# Use environment variables if set; otherwise replace with your actual values
$GATEWAY_URL = if ($env:GATEWAY_URL) { $env:GATEWAY_URL } else { "https://your-gateway.railway.app" }
$GATEWAY_API_KEY = if ($env:GATEWAY_API_KEY) { $env:GATEWAY_API_KEY } else { "your-gateway-api-key-here" }

if (-not $GATEWAY_API_KEY) {
    Write-Host "[ERROR] GATEWAY_API_KEY not set!" -ForegroundColor Red
    Write-Host "Run: `$env:GATEWAY_API_KEY = 'your-key-here'" -ForegroundColor Yellow
    exit 1
}

# Create a long system prompt (>1024 chars) to trigger constitution injection
$longSystemPrompt = @"
You are an expert software engineer working on a production system.
Follow these guidelines:
1. Write clean, maintainable code with proper error handling
2. Use type hints and documentation
3. Follow language-specific best practices
4. Prefer composition over inheritance
5. Keep functions focused and single-purpose
6. Optimize for readability and maintainability
7. Consider edge cases and error conditions
8. Write comprehensive tests for critical paths
9. Use descriptive variable and function names
10. Follow the principle of least surprise
11. Document complex logic and business rules
12. Use consistent formatting and style
13. Avoid premature optimization
14. Consider performance implications
15. Think about security from the start
16. Make code reviewable and understandable
17. Use version control best practices
18. Write self-documenting code when possible
19. Add comments only when necessary to explain why, not what
20. Keep dependencies minimal and well-justified
This system prompt is intentionally long to trigger Anthropic prompt caching with the platform constitution blocks for maximum cost savings.
"@

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "AI Gateway Cost Savings Test" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[INFO] Gateway URL: $GATEWAY_URL" -ForegroundColor Gray
Write-Host "[INFO] System prompt length: $($longSystemPrompt.Length) chars" -ForegroundColor Gray
Write-Host ""

# Test 1: First request (cache miss expected)
Write-Host "[TEST 1] Sending first request (cache MISS expected)..." -ForegroundColor Yellow

$body = @{
    model = "claude-sonnet-4-0"
    messages = @(
        @{
            role = "user"
            content = "Say 'Hello from AI Gateway cost savings test!'"
        }
    )
    system = $longSystemPrompt
    max_tokens = 100
} | ConvertTo-Json -Depth 10

$headers = @{
    "Authorization" = "Bearer $GATEWAY_API_KEY"
    "Content-Type" = "application/json"
}

try {
    $response1 = Invoke-RestMethod -Uri "$GATEWAY_URL/v1/chat/completions" -Method Post -Headers $headers -Body $body
    Write-Host "[SUCCESS] First request completed" -ForegroundColor Green
    Write-Host "Response: $($response1.choices[0].message.content)" -ForegroundColor White
} catch {
    Write-Host "[ERROR] First request failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Start-Sleep -Seconds 2

# Test 2: Second identical request (cache hit expected)
Write-Host "[TEST 2] Sending identical request (cache HIT expected for constitution blocks)..." -ForegroundColor Yellow

try {
    $response2 = Invoke-RestMethod -Uri "$GATEWAY_URL/v1/chat/completions" -Method Post -Headers $headers -Body $body
    Write-Host "[SUCCESS] Second request completed" -ForegroundColor Green
    Write-Host "Response: $($response2.choices[0].message.content)" -ForegroundColor White
} catch {
    Write-Host "[ERROR] Second request failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Checking Metrics for Savings..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

try {
    $metrics = Invoke-RestMethod -Uri "$GATEWAY_URL/metrics" -Method Get
    
    # Extract relevant metrics
    $tokensSaved = $metrics | Select-String -Pattern "gateway_tokens_saved_total" -AllMatches
    $cacheHits = $metrics | Select-String -Pattern "gateway_cache_hits_total" -AllMatches
    $promptCache = $metrics | Select-String -Pattern "gateway_prompt_cache_tokens" -AllMatches
    
    Write-Host ""
    Write-Host "Token Savings Metrics:" -ForegroundColor Green
    if ($tokensSaved) {
        $tokensSaved | ForEach-Object { Write-Host "  $_" -ForegroundColor White }
    } else {
        Write-Host "  (No token savings recorded yet)" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "Cache Hit Metrics:" -ForegroundColor Green
    if ($cacheHits) {
        $cacheHits | ForEach-Object { Write-Host "  $_" -ForegroundColor White }
    } else {
        Write-Host "  (No cache hits recorded yet)" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "Prompt Cache Metrics:" -ForegroundColor Green
    if ($promptCache) {
        $promptCache | ForEach-Object { Write-Host "  $_" -ForegroundColor White }
    } else {
        Write-Host "  (No prompt cache metrics recorded yet)" -ForegroundColor Gray
    }
    
} catch {
    Write-Host "[WARNING] Could not fetch metrics: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "Metrics endpoint may require authentication or may not be exposed." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Test Complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "What to look for:" -ForegroundColor Yellow
Write-Host "1. Second request should be faster (prompt cache hit)" -ForegroundColor White
Write-Host "2. Check Railway logs for 'Gateway reduction' messages" -ForegroundColor White
Write-Host "3. Anthropic response should show cache_read_input_tokens > 0 on 2nd request" -ForegroundColor White
Write-Host "4. Metrics should show gateway_prompt_cache_tokens_total increasing" -ForegroundColor White
Write-Host ""
