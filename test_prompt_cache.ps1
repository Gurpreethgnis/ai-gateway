# Test Anthropic Prompt Cache (different messages, same system prompt)
# This shows cache_read_input_tokens when the system prompt is reused

$GATEWAY_URL = if ($env:GATEWAY_URL) { $env:GATEWAY_URL } else { "https://cursor.gursimanoor.com" }
$GATEWAY_API_KEY = if ($env:GATEWAY_API_KEY) { $env:GATEWAY_API_KEY } else { "your-key" }

# Long system prompt (>1024 chars) that will trigger constitution injection and caching
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

$headers = @{
    "Authorization" = "Bearer $GATEWAY_API_KEY"
    "Content-Type" = "application/json"
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Anthropic Prompt Cache Test" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[INFO] System prompt: $($longSystemPrompt.Length) chars" -ForegroundColor Gray
Write-Host ""

# Request 1 - creates prompt cache
Write-Host "[TEST 1] First request (will CREATE prompt cache)..." -ForegroundColor Yellow
$body1 = @{
    model = "claude-sonnet-4-0"
    messages = @(
        @{ role = "user"; content = "Say hello" }
    )
    system = $longSystemPrompt
    max_tokens = 50
} | ConvertTo-Json -Depth 10

try {
    $response1 = Invoke-RestMethod -Uri "$GATEWAY_URL/v1/chat/completions" -Method Post -Headers $headers -Body $body1
    Write-Host "[SUCCESS] Response: $($response1.choices[0].message.content)" -ForegroundColor Green
    if ($response1.usage.cache_creation_input_tokens) {
        Write-Host "  Cache created: $($response1.usage.cache_creation_input_tokens) tokens" -ForegroundColor Cyan
    }
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Start-Sleep -Seconds 2

# Request 2 - should READ from prompt cache (different message, same system)
Write-Host "[TEST 2] Second request with DIFFERENT message (should READ from cache)..." -ForegroundColor Yellow
$body2 = @{
    model = "claude-sonnet-4-0"
    messages = @(
        @{ role = "user"; content = "Say goodbye" }
    )
    system = $longSystemPrompt
    max_tokens = 50
} | ConvertTo-Json -Depth 10

try {
    $response2 = Invoke-RestMethod -Uri "$GATEWAY_URL/v1/chat/completions" -Method Post -Headers $headers -Body $body2
    Write-Host "[SUCCESS] Response: $($response2.choices[0].message.content)" -ForegroundColor Green
    if ($response2.usage.cache_read_input_tokens) {
        Write-Host "  Cache read: $($response2.usage.cache_read_input_tokens) tokens (saved ~90%!)" -ForegroundColor Green
    }
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Check /admin/metrics for gateway_prompt_cache_tokens_total" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
