# Simple test to see Anthropic prompt cache savings in the response
$GATEWAY_URL = if ($env:GATEWAY_URL) { $env:GATEWAY_URL } else { "https://cursor.gursimanoor.com" }
$GATEWAY_API_KEY = if ($env:GATEWAY_API_KEY) { $env:GATEWAY_API_KEY } else { "your-key" }

# Long system prompt (>1024 chars to trigger caching)
$longSystemPrompt = @"
You are an expert software engineer. Follow these guidelines:
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
This system prompt is intentionally long to trigger Anthropic prompt caching for maximum cost savings.
"@

$headers = @{
    "Authorization" = "Bearer $GATEWAY_API_KEY"
    "Content-Type" = "application/json"
    "X-No-Cache" = "1"  # Bypass response cache to test Anthropic prompt cache
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Testing Anthropic Prompt Cache Savings" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Request 1 - First time (creates cache)
$timestamp1 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
Write-Host "[Request 1] Sending first request (timestamp: $timestamp1)..." -ForegroundColor Yellow
$body1 = @{
    model = "claude-sonnet-4-0"
    messages = @(
        @{ role = "user"; content = "Just say 'Hello' and nothing else. Time: $timestamp1" }
    )
    system = $longSystemPrompt
    max_tokens = 20
} | ConvertTo-Json -Depth 10

try {
    $response1 = Invoke-RestMethod -Uri "$GATEWAY_URL/v1/chat/completions" -Method Post -Headers $headers -Body $body1
    Write-Host "[SUCCESS] Response: $($response1.choices[0].message.content)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Cyan
    Write-Host "  prompt_tokens: $($response1.usage.prompt_tokens)" -ForegroundColor White
    Write-Host "  completion_tokens: $($response1.usage.completion_tokens)" -ForegroundColor White
    Write-Host "  cache_creation_input_tokens: $($response1.usage.cache_creation_input_tokens)" -ForegroundColor Yellow
    Write-Host "  cache_read_input_tokens: $($response1.usage.cache_read_input_tokens)" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Start-Sleep -Seconds 3

# Request 2 - Different message, should hit cache
$timestamp2 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
Write-Host "[Request 2] Sending second request with different message (timestamp: $timestamp2)..." -ForegroundColor Yellow
$body2 = @{
    model = "claude-sonnet-4-0"
    messages = @(
        @{ role = "user"; content = "Just say 'Goodbye' and nothing else. Time: $timestamp2" }
    )
    system = $longSystemPrompt
    max_tokens = 20
} | ConvertTo-Json -Depth 10

try {
    $response2 = Invoke-RestMethod -Uri "$GATEWAY_URL/v1/chat/completions" -Method Post -Headers $headers -Body $body2
    Write-Host "[SUCCESS] Response: $($response2.choices[0].message.content)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Cyan
    Write-Host "  prompt_tokens: $($response2.usage.prompt_tokens)" -ForegroundColor White
    Write-Host "  completion_tokens: $($response2.usage.completion_tokens)" -ForegroundColor White
    Write-Host "  cache_creation_input_tokens: $($response2.usage.cache_creation_input_tokens)" -ForegroundColor Yellow
    Write-Host "  cache_read_input_tokens: $($response2.usage.cache_read_input_tokens) <-- YOUR SAVINGS!" -ForegroundColor Green
    
    if ($response2.usage.cache_read_input_tokens -gt 0) {
        $savingsPct = ($response2.usage.cache_read_input_tokens / $response2.usage.prompt_tokens) * 100
        Write-Host ""
        Write-Host "  💰 SAVINGS: ~$([math]::Round($savingsPct, 1))% of prompt tokens read from cache!" -ForegroundColor Green
    }
    Write-Host ""
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Next: Check /admin/metrics for gateway_prompt_cache_tokens_total" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
