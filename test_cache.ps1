$ErrorActionPreference = "Stop"

$apiKey = "bv7CZqbgqeWkS67+3Yl/N1bBWku8T7QPfZyNxfUKqMdzQG+ZTpwN0OXAZwORt4IL"
$url = "https://cursor.gursimanoor.com/v1/chat/completions"

# Create a huge dummy system prompt (> 1024 tokens)
$hugeSystem = "You are a helpful coding assistant. " * 300

# Build the payload as a PowerShell hashtable so we never have to deal with quote escaping
$payloadObj = @{
    model = "claude-3-5-sonnet-20241022"
    messages = @(
        @{
            role = "system"
            content = $hugeSystem
        },
        @{
            role = "user"
            content = "Hi, what is 2+2?"
        }
    )
    max_tokens = 50
    stream = $false
}

# Convert to clean JSON string natively
$jsonPayload = $payloadObj | ConvertTo-Json -Depth 5

$headers = @{
    "Content-Type" = "application/json"
    "Authorization" = "Bearer $apiKey"
}

Write-Host ""
Write-Host "=== TEST 1: First Request (Should WRITE to Anthropic Cache) ==="
try {
    $response1 = Invoke-RestMethod -Uri $url -Method Post -Headers $headers -Body $jsonPayload
    Write-Host "Success! Response Choice:"
    Write-Host $response1.choices[0].message.content
    
    Write-Host ""
    Write-Host "USAGE STATS:"
    Write-Host "- Total Tokens: $($response1.usage.total_tokens)"
    Write-Host "- Input Tokens Billed: $($response1.usage.prompt_tokens)"
    if ($null -ne $response1.usage.prompt_tokens_details) {
        Write-Host "- Cached Tokens: $($response1.usage.prompt_tokens_details.cached_tokens)"
    }
} catch {
    Write-Host "Request 1 Failed:"
    Write-Host $_.Exception.Response.StatusCode
    $errStream = $_.Exception.Response.GetResponseStream()
    $reader = New-Object System.IO.StreamReader($errStream)
    Write-Host $reader.ReadToEnd()
}

Write-Host ""
Write-Host "=== Waiting 3 seconds... ==="
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "=== TEST 2: Second Request (Should READ from Anthropic Cache) ==="
try {
    $response2 = Invoke-RestMethod -Uri $url -Method Post -Headers $headers -Body $jsonPayload
    Write-Host "Success! Response Choice:"
    Write-Host $response2.choices[0].message.content
    
    Write-Host ""
    Write-Host "USAGE STATS:"
    Write-Host "- Total Tokens: $($response2.usage.total_tokens)"
    Write-Host "- Input Tokens Billed: $($response2.usage.prompt_tokens)"
    if ($null -ne $response2.usage.prompt_tokens_details) {
        Write-Host "- Cached Tokens: $($response2.usage.prompt_tokens_details.cached_tokens)"
    }
} catch {
    Write-Host "Request 2 Failed:"
    Write-Host $_.Exception.Response.StatusCode
    $errStream = $_.Exception.Response.GetResponseStream()
    $reader = New-Object System.IO.StreamReader($errStream)
    Write-Host $reader.ReadToEnd()
}
Write-Host ""
