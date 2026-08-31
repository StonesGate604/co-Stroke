param(
    [switch]$IncludeAllFormalCheckpoints,
    [switch]$SkipQuickDraw,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repository = "StonesGate604/co-Stroke"
$quickDrawUrl = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/cat.ndjson"

$formalCheckpoints = @(
    [pscustomobject]@{
        Version = "v0.3.1"
        Destination = "runs/stroke5-transformer-v3-cat/checkpoint.pt"
        Sha256 = "996F8B0DF3187FD3A5B0BCF079D65675A30A8C26EEED80207CE3627B3F8EE0E3"
    },
    [pscustomobject]@{
        Version = "v0.4.0.1"
        Destination = "runs/stroke-relational-v4-cat/checkpoint.pt"
        Sha256 = "A83E234C0C1E2572B5AA5AF9044C8CBB9FA83F9E44CA67DC98D54CEC438312CC"
    },
    [pscustomobject]@{
        Version = "v0.4.1"
        Destination = "runs/stroke-multimodal-v41-cat/checkpoint.pt"
        Sha256 = "A925CAA81F00A342033CB7E8ED47164AF4972AC11A9CE4C66108796CCC124C80"
    }
)

function Resolve-ProjectPath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $projectRoot = Split-Path -Parent $PSScriptRoot
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $RelativePath))
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Get-GitHubHeaders {
    $credentialInput = "protocol=https`nhost=github.com`n`n"
    $credentialOutput = $credentialInput | git credential fill
    $tokenLine = $credentialOutput |
        Where-Object { $_ -like "password=*" } |
        Select-Object -First 1

    if (-not $tokenLine) {
        throw "No GitHub credential was found. Sign in to GitHub for Git first, then retry."
    }

    $token = $tokenLine.Substring("password=".Length)
    return @{
        Authorization = "Bearer $token"
        Accept = "application/vnd.github+json"
        "X-GitHub-Api-Version" = "2022-11-28"
        "User-Agent" = "co-Stroke-asset-sync"
    }
}

function Download-Asset {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string]$ExpectedSha256 = "",
        [hashtable]$Headers = @{}
    )

    $absoluteDestination = Resolve-ProjectPath $Destination
    $destinationDirectory = Split-Path -Parent $absoluteDestination
    New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null

    if (Test-Path -LiteralPath $absoluteDestination) {
        if (-not $Force -and $ExpectedSha256 -and (Get-Sha256 $absoluteDestination) -eq $ExpectedSha256) {
            Write-Host "Already current: $Destination"
            return
        }

        if (-not $ExpectedSha256 -and -not $Force) {
            Write-Host "Already present: $Destination"
            return
        }

        if (-not $Force) {
            throw "Existing file differs: $Destination. Re-run with -Force to replace it."
        }
    }

    $temporaryPath = "$absoluteDestination.download"
    try {
        Write-Host "Downloading: $Destination"
        Invoke-WebRequest -Uri $Url -Headers $Headers -OutFile $temporaryPath

        if ($ExpectedSha256) {
            $actualSha256 = Get-Sha256 $temporaryPath
            if ($actualSha256 -ne $ExpectedSha256) {
                throw "SHA-256 mismatch for $Destination. Expected $ExpectedSha256, got $actualSha256."
            }
        }

        Move-Item -Force -LiteralPath $temporaryPath -Destination $absoluteDestination
        Write-Host "Ready: $Destination"
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath) {
            Remove-Item -Force -LiteralPath $temporaryPath
        }
    }
}

if (-not $SkipQuickDraw) {
    Download-Asset `
        -Url $quickDrawUrl `
        -Destination "data/quickdraw/cat.ndjson"
}

$selectedCheckpoints = if ($IncludeAllFormalCheckpoints) {
    $formalCheckpoints
}
else {
    $formalCheckpoints | Where-Object Version -eq "v0.4.1"
}

foreach ($checkpoint in $selectedCheckpoints) {
    $githubHeaders = Get-GitHubHeaders
    $releaseApiUrl = "https://api.github.com/repos/$repository/releases/tags/$($checkpoint.Version)"
    $release = Invoke-RestMethod -Uri $releaseApiUrl -Headers $githubHeaders
    $releaseAsset = $release.assets |
        Where-Object { $_.name -eq "checkpoint.pt" } |
        Select-Object -First 1

    if (-not $releaseAsset) {
        throw "Release $($checkpoint.Version) does not contain checkpoint.pt."
    }

    $downloadHeaders = $githubHeaders.Clone()
    $downloadHeaders.Accept = "application/octet-stream"
    Download-Asset `
        -Url $releaseAsset.url `
        -Destination $checkpoint.Destination `
        -ExpectedSha256 $checkpoint.Sha256 `
        -Headers $downloadHeaders
}

Write-Host "Asset synchronization complete."
