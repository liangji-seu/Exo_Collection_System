[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipSmokeChecks,
    [switch]$SkipOptionalInstaller,
    [switch]$AllowDirtyWorkingTree
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$SourceRoot = Join-Path $ProjectRoot "src"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$CollectorSpec = Join-Path $PSScriptRoot "collector.spec"
$DataStudioSpec = Join-Path $PSScriptRoot "data_studio.spec"
$DistRoot = Join-Path $ProjectRoot "dist"
$CollectorExe = Join-Path $DistRoot "ExoCollector.exe"
$DataStudioExe = Join-Path $DistRoot "ExoDataStudio.exe"
$ReleaseRoot = Join-Path $ProjectRoot "release"
$InstallerSpec = Join-Path $PSScriptRoot "installer\ExoCollectionSystem.iss"
$StartHere = Join-Path $PSScriptRoot "installer\README_START_HERE.txt"

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $Utf8NoBom)
}

function Assert-NoReparsePointInProjectChildPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $ProjectFullPath = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\')
    $Candidate = [System.IO.Path]::GetFullPath($Path)
    $ProjectPrefix = $ProjectFullPath + '\'
    if (-not $Candidate.StartsWith($ProjectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing a destructive operation outside the project root: $Candidate"
    }

    $RelativePath = $Candidate.Substring($ProjectPrefix.Length)
    $Cursor = $ProjectFullPath
    foreach ($Segment in ($RelativePath -split '[\\/]')) {
        if ([string]::IsNullOrWhiteSpace($Segment)) {
            continue
        }
        $Cursor = Join-Path $Cursor $Segment
        if (-not (Test-Path -LiteralPath $Cursor)) {
            continue
        }
        $Item = Get-Item -Force -LiteralPath $Cursor
        if (($Item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing a destructive operation through a reparse point: $Cursor"
        }
    }
}

function Assert-SafeProjectChildPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $ProjectPrefix = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\') + '\'
    $Candidate = [System.IO.Path]::GetFullPath($Path)
    if (-not $Candidate.StartsWith($ProjectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing a destructive operation outside the project root: $Candidate"
    }
    Assert-NoReparsePointInProjectChildPath -Path $Candidate
    return $Candidate
}

function Get-GitRepositoryState {
    param(
        [Parameter(Mandatory = $true)]
        [AllowNull()]
        [System.Management.Automation.ApplicationInfo]$GitCommand
    )

    if ($null -eq $GitCommand) {
        return [pscustomobject]@{
            available = $false
            commit = "unknown-local-build"
            dirty = $null
            status = @()
        }
    }

    $InsideOutput = & $GitCommand.Source -C $ProjectRoot rev-parse --is-inside-work-tree 2>$null
    if ($LASTEXITCODE -ne 0 -or ([string]($InsideOutput | Select-Object -Last 1)).Trim() -ne "true") {
        return [pscustomobject]@{
            available = $false
            commit = "unknown-local-build"
            dirty = $null
            status = @()
        }
    }

    $CommitOutput = & $GitCommand.Source -C $ProjectRoot rev-parse --verify HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Git repository was detected, but HEAD could not be resolved."
    }
    $Commit = ([string]($CommitOutput | Select-Object -Last 1)).Trim()
    if ($Commit -notmatch '^[0-9a-fA-F]{40}$') {
        throw "Git returned an invalid HEAD commit: $Commit"
    }

    $StatusOutput = @(& $GitCommand.Source -C $ProjectRoot status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect the Git working tree."
    }
    $StatusLines = @(
        $StatusOutput |
            ForEach-Object { [string]$_ } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    return [pscustomobject]@{
        available = $true
        commit = $Commit.ToLowerInvariant()
        dirty = ($StatusLines.Count -gt 0)
        status = $StatusLines
    }
}

function Assert-GitReleaseStateUnchanged {
    param(
        [Parameter(Mandatory = $true)]
        [object]$InitialState,
        [Parameter(Mandatory = $true)]
        [AllowNull()]
        [System.Management.Automation.ApplicationInfo]$GitCommand
    )

    if (-not $InitialState.available) {
        return
    }
    $CurrentState = Get-GitRepositoryState -GitCommand $GitCommand
    if (-not $CurrentState.available -or $CurrentState.commit -ne $InitialState.commit) {
        throw "Git HEAD changed while the release was being built; discard this build and run it again."
    }
    if (-not $AllowDirtyWorkingTree -and $CurrentState.dirty) {
        throw "The Git working tree changed while the release was being built; discard this build and run it again."
    }
}

function Get-StreamSha256Hex {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.Stream]$Stream
    )

    $Sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $HashBytes = $Sha256.ComputeHash($Stream)
        return ([System.BitConverter]::ToString($HashBytes)).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $Sha256.Dispose()
    }
}

function Test-ZipBundle {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ZipPath,
        [Parameter(Mandatory = $true)]
        [string]$TopLevelDirectory,
        [Parameter(Mandatory = $true)]
        [object[]]$ManifestFiles,
        [Parameter(Mandatory = $true)]
        [string]$BuildManifestPath
    )

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Archive = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        $ExpectedPaths = @{}
        foreach ($File in $ManifestFiles) {
            $EntryPath = "$TopLevelDirectory/$($File.path)"
            $ExpectedPaths[$EntryPath] = $File
        }
        $ManifestEntryPath = "$TopLevelDirectory/BUILD_MANIFEST.json"
        $ExpectedPaths[$ManifestEntryPath] = [pscustomobject]@{
            size_bytes = (Get-Item -LiteralPath $BuildManifestPath).Length
            sha256 = (Get-FileHash -LiteralPath $BuildManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
        }

        $SeenPaths = @{}
        foreach ($Entry in $Archive.Entries) {
            if ([string]::IsNullOrEmpty($Entry.Name)) {
                continue
            }
            $EntryPath = $Entry.FullName.Replace('\', '/')
            if ($SeenPaths.ContainsKey($EntryPath)) {
                throw "ZIP bundle contains a duplicate entry: $EntryPath"
            }
            $SeenPaths[$EntryPath] = $true
            if (-not $ExpectedPaths.ContainsKey($EntryPath)) {
                throw "ZIP bundle contains an unexpected entry: $EntryPath"
            }
            $ExpectedFile = $ExpectedPaths[$EntryPath]
            if ([long]$Entry.Length -ne [long]$ExpectedFile.size_bytes) {
                throw "ZIP entry size mismatch: $EntryPath"
            }
            $EntryStream = $Entry.Open()
            try {
                $ArchiveHash = Get-StreamSha256Hex -Stream $EntryStream
            }
            finally {
                $EntryStream.Dispose()
            }
            if ($ArchiveHash -ne $ExpectedFile.sha256) {
                throw "ZIP entry SHA-256 mismatch: $EntryPath"
            }
        }

        foreach ($ExpectedPath in $ExpectedPaths.Keys) {
            if (-not $SeenPaths.ContainsKey($ExpectedPath)) {
                throw "ZIP bundle is missing an expected entry: $ExpectedPath"
            }
        }
    }
    finally {
        $Archive.Dispose()
    }
}

function Invoke-NativeStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host $Label
    & $FilePath @Arguments
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "$Label failed with exit code $ExitCode."
    }
}

function Invoke-FrozenSmokeCheck {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [int]$TimeoutSeconds = 45
    )

    Write-Host $Label
    $QtPlatformWasSet = Test-Path -LiteralPath "Env:\QT_QPA_PLATFORM"
    $PreviousQtPlatform = $env:QT_QPA_PLATFORM
    $env:QT_QPA_PLATFORM = "offscreen"
    $Process = $null
    try {
        $Process = Start-Process `
            -FilePath $Executable `
            -ArgumentList $Arguments `
            -WorkingDirectory $ProjectRoot `
            -WindowStyle Hidden `
            -PassThru
        if (-not $Process.WaitForExit($TimeoutSeconds * 1000)) {
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
            throw "$Label timed out after $TimeoutSeconds seconds."
        }
        $Process.Refresh()
        if ($Process.ExitCode -ne 0) {
            throw "$Label failed with exit code $($Process.ExitCode)."
        }
    }
    finally {
        if ($null -ne $Process) {
            $Process.Dispose()
        }
        if ($QtPlatformWasSet) {
            $env:QT_QPA_PLATFORM = $PreviousQtPlatform
        }
        else {
            Remove-Item -LiteralPath "Env:\QT_QPA_PLATFORM" -ErrorAction SilentlyContinue
        }
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw @"
Python virtual environment was not found at:
$Python

On a new Windows computer run this zero-argument entry point first:
  .\First_Time_Setup.cmd
"@
}

foreach ($RequiredFile in @(
    $CollectorSpec,
    $DataStudioSpec,
    $InstallerSpec,
    $StartHere,
    (Join-Path $ProjectRoot "Run_ExoCollector.cmd"),
    (Join-Path $ProjectRoot "Run_ExoDataStudio.cmd"),
    (Join-Path $ProjectRoot "README.md")
)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required build input was not found: $RequiredFile"
    }
}

$RuntimeJsonOutput = & $Python -c "import json,platform,struct,sys; print(json.dumps({'version': f'{sys.version_info.major}.{sys.version_info.minor}', 'bits': struct.calcsize('P') * 8, 'platform': platform.system()}))"
$RuntimeExitCode = $LASTEXITCODE
if ($RuntimeExitCode -ne 0) {
    throw "Unable to inspect the virtual-environment Python runtime (exit code $RuntimeExitCode)."
}
$Runtime = ([string]($RuntimeJsonOutput | Select-Object -Last 1)).Trim() | ConvertFrom-Json
if ($Runtime.version -ne "3.11") {
    throw "This project requires Python 3.11, but .venv uses Python $($Runtime.version)."
}
if ([int]$Runtime.bits -ne 64 -or $Runtime.platform -ne "Windows") {
    throw "Windows packaging requires 64-bit Windows Python; detected $($Runtime.platform) $($Runtime.bits)-bit."
}

$PyInstallerOutput = & $Python -c "import PyInstaller; print(PyInstaller.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller is unavailable. Run .\First_Time_Setup.cmd to install development and packaging dependencies.'
}
$PyInstallerVersion = ([string]($PyInstallerOutput | Select-Object -Last 1)).Trim()

$PackageInfoOutput = & $Python -c "import exo_collection,json; print(json.dumps({'version': exo_collection.__version__, 'file': exo_collection.__file__}, ensure_ascii=True))"
if ($LASTEXITCODE -ne 0) {
    throw "The exo_collection package is not importable from the current .venv. Run .\First_Time_Setup.cmd."
}
$PackageInfo = ([string]($PackageInfoOutput | Select-Object -Last 1)).Trim() | ConvertFrom-Json
$ApplicationVersion = ([string]$PackageInfo.version).Trim()
if ($ApplicationVersion -notmatch '^\d+\.\d+\.\d+$') {
    throw "Application version is not semantic x.y.z: $ApplicationVersion"
}
$PackagePath = [System.IO.Path]::GetFullPath([string]$PackageInfo.file)
$SourcePrefix = [System.IO.Path]::GetFullPath($SourceRoot).TrimEnd('\') + '\'
if (-not $PackagePath.StartsWith($SourcePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw @"
The active .venv imports exo_collection from a different checkout:
  $PackagePath

Expected a path under:
  $SourceRoot

Run .\First_Time_Setup.cmd in this checkout before packaging it.
"@
}

$GitCommand = Get-Command "git.exe" -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
$InitialGitState = Get-GitRepositoryState -GitCommand $GitCommand
if (-not $InitialGitState.available) {
    if (-not $AllowDirtyWorkingTree) {
        throw @"
A verified Git checkout is required for a default release build so the
executable and BUILD_MANIFEST.json have complete source provenance.

Install Git and build from a cloned repository. For a non-release developer
probe only, rerun packaging\build_windows.ps1 with -AllowDirtyWorkingTree;
the manifest will mark source provenance as unavailable.
"@
    }
    Write-Warning "Git provenance is unavailable. This explicitly permitted developer build will be marked non-reproducible."
}
elseif ($InitialGitState.dirty) {
    if (-not $AllowDirtyWorkingTree) {
        $StatusPreview = ($InitialGitState.status | Select-Object -First 12) -join [Environment]::NewLine
        throw @"
The Git working tree is not clean. A default release cannot truthfully claim
that its executable bytes correspond to commit $($InitialGitState.commit).

$StatusPreview

Commit or otherwise resolve the changes, then run Build_Windows.cmd again.
For a non-release developer probe only, invoke packaging\build_windows.ps1
with -AllowDirtyWorkingTree; BUILD_MANIFEST.json will record dirty=true.
"@
    }
    Write-Warning "The Git working tree is dirty. This explicitly permitted developer build will record dirty=true and is not reproducible from the named commit alone."
}

Write-Host "Windows release build: Exo Collection System $ApplicationVersion"
Write-Host "Python $($Runtime.version) $($Runtime.bits)-bit; PyInstaller $PyInstallerVersion"
if ($InitialGitState.available) {
    Write-Host "Source commit $($InitialGitState.commit); dirty=$($InitialGitState.dirty)"
}

$BuildGitCommitWasSet = Test-Path -LiteralPath "Env:\EXO_BUILD_GIT_COMMIT"
$PreviousBuildGitCommit = $env:EXO_BUILD_GIT_COMMIT
$BuildGitDirtyWasSet = Test-Path -LiteralPath "Env:\EXO_BUILD_GIT_DIRTY"
$PreviousBuildGitDirty = $env:EXO_BUILD_GIT_DIRTY
$BuildAppVersionWasSet = Test-Path -LiteralPath "Env:\EXO_BUILD_APP_VERSION"
$PreviousBuildAppVersion = $env:EXO_BUILD_APP_VERSION
$env:EXO_BUILD_GIT_COMMIT = [string]$InitialGitState.commit
$env:EXO_BUILD_GIT_DIRTY = if ($null -eq $InitialGitState.dirty) { "unknown" } else { ([bool]$InitialGitState.dirty).ToString().ToLowerInvariant() }
$env:EXO_BUILD_APP_VERSION = $ApplicationVersion

Push-Location -LiteralPath $ProjectRoot
try {
    if (-not $SkipTests) {
        Invoke-NativeStep `
            -Label "[1/8] Running the complete automated test suite" `
            -FilePath $Python `
            -Arguments @("-m", "pytest")
    }
    else {
        Write-Warning "[1/8] Automated tests were explicitly skipped for this developer build."
    }

    Invoke-NativeStep `
        -Label "[2/8] Building ExoCollector.exe" `
        -FilePath $Python `
        -Arguments @("-m", "PyInstaller", "--noconfirm", "--clean", $CollectorSpec)

    Invoke-NativeStep `
        -Label "[3/8] Building ExoDataStudio.exe" `
        -FilePath $Python `
        -Arguments @("-m", "PyInstaller", "--noconfirm", "--clean", $DataStudioSpec)
}
finally {
    Pop-Location
    if ($BuildGitCommitWasSet) {
        $env:EXO_BUILD_GIT_COMMIT = $PreviousBuildGitCommit
    }
    else {
        Remove-Item -LiteralPath "Env:\EXO_BUILD_GIT_COMMIT" -ErrorAction SilentlyContinue
    }
    if ($BuildGitDirtyWasSet) {
        $env:EXO_BUILD_GIT_DIRTY = $PreviousBuildGitDirty
    }
    else {
        Remove-Item -LiteralPath "Env:\EXO_BUILD_GIT_DIRTY" -ErrorAction SilentlyContinue
    }
    if ($BuildAppVersionWasSet) {
        $env:EXO_BUILD_APP_VERSION = $PreviousBuildAppVersion
    }
    else {
        Remove-Item -LiteralPath "Env:\EXO_BUILD_APP_VERSION" -ErrorAction SilentlyContinue
    }
}

foreach ($Executable in @($CollectorExe, $DataStudioExe)) {
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "PyInstaller completed without the expected executable: $Executable"
    }
    if ((Get-Item -LiteralPath $Executable).Length -le 0) {
        throw "PyInstaller produced an empty executable: $Executable"
    }
}

# Verify that the generated build-info.json faithfully records the Git state
# that the spec file captured at freeze time.
$BuildInfoPath = Join-Path $ProjectRoot "build\build-info.json"
if (-not (Test-Path -LiteralPath $BuildInfoPath -PathType Leaf)) {
    throw "build-info.json was not generated by the PyInstaller spec file: $BuildInfoPath"
}
$BuildInfoJson = Get-Content -LiteralPath $BuildInfoPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]$BuildInfoJson.git_commit -ne $InitialGitState.commit) {
    throw "build-info.json commit ($($BuildInfoJson.git_commit)) does not match the build-time HEAD ($($InitialGitState.commit))."
}
if ($null -ne $InitialGitState.dirty) {
    $RecordedDirty = $BuildInfoJson.git_worktree_dirty
    if ($null -eq $RecordedDirty) {
        throw "build-info.json has no dirty flag; expected $($InitialGitState.dirty)."
    }
    $RecordedDirty = [System.Convert]::ToBoolean($RecordedDirty)
    if ($RecordedDirty -ne $InitialGitState.dirty) {
        throw "build-info.json dirty flag ($($BuildInfoJson.git_worktree_dirty)) does not match expected ($($InitialGitState.dirty))."
    }
}
Write-Host "build-info.json provenance verified: commit=$($BuildInfoJson.git_commit), dirty=$($BuildInfoJson.git_worktree_dirty)"

if (-not $SkipSmokeChecks) {
    Invoke-FrozenSmokeCheck `
        -Label "[4/8] Offscreen startup check: ExoCollector.exe" `
        -Executable $CollectorExe `
        -Arguments @("--smoke-test") `
        -TimeoutSeconds 45
    Invoke-FrozenSmokeCheck `
        -Label "[5/8] Spawned simulated collection check: ExoCollector.exe" `
        -Executable $CollectorExe `
        -Arguments @("--collect-smoke-test", "--duration", "0.25") `
        -TimeoutSeconds 90
    Invoke-FrozenSmokeCheck `
        -Label "[6/8] Offscreen startup/Catalog check: ExoDataStudio.exe" `
        -Executable $DataStudioExe `
        -Arguments @("--smoke-test") `
        -TimeoutSeconds 45
}
else {
    Write-Warning "[4-6/8] Frozen-application smoke checks were explicitly skipped."
}

Write-Host "[7/8] Creating one distributable ZIP bundle"
Assert-GitReleaseStateUnchanged -InitialState $InitialGitState -GitCommand $GitCommand
$BundleName = "ExoCollectionSystem-$ApplicationVersion-windows-x64"
$BundleStage = Assert-SafeProjectChildPath (Join-Path $ReleaseRoot $BundleName)
$BundleZip = Assert-SafeProjectChildPath (Join-Path $ReleaseRoot "$BundleName.zip")
$BundleZipSha = Assert-SafeProjectChildPath "$BundleZip.sha256"
if (Test-Path -LiteralPath $BundleStage) {
    Remove-Item -LiteralPath $BundleStage -Recurse -Force
}
if (Test-Path -LiteralPath $BundleZip) {
    Remove-Item -LiteralPath $BundleZip -Force
}
if (Test-Path -LiteralPath $BundleZipSha) {
    Remove-Item -LiteralPath $BundleZipSha -Force
}
$BundleDist = Join-Path $BundleStage "dist"
New-Item -ItemType Directory -Path $BundleDist -Force | Out-Null
$PayloadSources = @(
    [pscustomobject]@{ source = $CollectorExe; path = "dist/ExoCollector.exe" },
    [pscustomobject]@{ source = $DataStudioExe; path = "dist/ExoDataStudio.exe" },
    [pscustomobject]@{ source = (Join-Path $ProjectRoot "Run_ExoCollector.cmd"); path = "Run_ExoCollector.cmd" },
    [pscustomobject]@{ source = (Join-Path $ProjectRoot "Run_ExoDataStudio.cmd"); path = "Run_ExoDataStudio.cmd" },
    [pscustomobject]@{ source = $StartHere; path = "README_START_HERE.txt" },
    [pscustomobject]@{ source = (Join-Path $ProjectRoot "README.md"); path = "README_PROJECT.md" }
)
foreach ($Payload in $PayloadSources) {
    $Destination = Join-Path $BundleStage $Payload.path.Replace('/', '\')
    $DestinationParent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $DestinationParent -Force | Out-Null
    Copy-Item -LiteralPath $Payload.source -Destination $Destination
}

$ManifestFiles = @(
    foreach ($Payload in ($PayloadSources | Sort-Object path)) {
        $StagedPath = Join-Path $BundleStage $Payload.path.Replace('/', '\')
        [ordered]@{
            path = $Payload.path
            size_bytes = (Get-Item -LiteralPath $StagedPath).Length
            sha256 = (Get-FileHash -LiteralPath $StagedPath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
)

$DependencyVersionsOutput = & $Python -c "import importlib.metadata as m,json; names=['alembic','h5py','numpy','paramiko','pydantic','PySide6','pyqtgraph','scp','SQLAlchemy','pyinstaller']; print(json.dumps({name:m.version(name) for name in names}, sort_keys=True))"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to record exact build dependency versions."
}
$DependencyVersions = ([string]($DependencyVersionsOutput | Select-Object -Last 1)).Trim() | ConvertFrom-Json
$SourceProvenance = if (-not $InitialGitState.available) {
    "unavailable"
}
elseif ($InitialGitState.dirty) {
    "git-dirty-explicit-developer-build"
}
else {
    "git-clean"
}
$BuildManifest = [ordered]@{
    schema_version = "1.1.0"
    application_version = $ApplicationVersion
    git_commit = [string]$InitialGitState.commit
    git_worktree_dirty = $InitialGitState.dirty
    source_provenance = $SourceProvenance
    source_matches_git_commit = [bool]($InitialGitState.available -and -not $InitialGitState.dirty)
    built_at_utc = [DateTime]::UtcNow.ToString("o")
    target = "windows-x64"
    hardware_support = "simulated devices and replaceable adapter interfaces only"
    build_environment = [ordered]@{
        python_version = [string]$Runtime.version
        python_bits = [int]$Runtime.bits
        pyinstaller_version = $PyInstallerVersion
        dependencies = $DependencyVersions
    }
    verification = [ordered]@{
        automated_tests_skipped = [bool]$SkipTests
        frozen_smoke_checks_skipped = [bool]$SkipSmokeChecks
    }
    files = $ManifestFiles
}
$BuildManifestPath = Join-Path $BundleStage "BUILD_MANIFEST.json"
$BuildManifestJson = $BuildManifest | ConvertTo-Json -Depth 8
Write-Utf8NoBom -Path $BuildManifestPath -Content ($BuildManifestJson + [Environment]::NewLine)
New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
Compress-Archive -LiteralPath $BundleStage -DestinationPath $BundleZip -CompressionLevel Optimal
if (-not (Test-Path -LiteralPath $BundleZip -PathType Leaf)) {
    throw "ZIP bundle was not created: $BundleZip"
}
Test-ZipBundle `
    -ZipPath $BundleZip `
    -TopLevelDirectory $BundleName `
    -ManifestFiles $ManifestFiles `
    -BuildManifestPath $BuildManifestPath
$BundleZipHash = (Get-FileHash -LiteralPath $BundleZip -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Utf8NoBom `
    -Path $BundleZipSha `
    -Content "$BundleZipHash  $([System.IO.Path]::GetFileName($BundleZip))$([Environment]::NewLine)"

Write-Host "[8/8] Checking for optional Inno Setup compiler"
$InstallerOutput = $null
if (-not $SkipOptionalInstaller) {
    $IsccCandidates = @(
        (Get-Command "ISCC.exe" -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1 | ForEach-Object { $_.Source }),
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | Select-Object -Unique
    $Iscc = $IsccCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if ($null -ne $Iscc) {
        $InstallerOutput = Assert-SafeProjectChildPath (Join-Path $ReleaseRoot "ExoCollectionSystem-$ApplicationVersion-Setup.exe")
        if (Test-Path -LiteralPath $InstallerOutput) {
            Remove-Item -LiteralPath $InstallerOutput -Force
        }
        & $Iscc "/DAppVersion=$ApplicationVersion" "/O$ReleaseRoot" $InstallerSpec
        if ($LASTEXITCODE -ne 0) {
            throw "Inno Setup compilation failed with exit code $LASTEXITCODE."
        }
        if (-not (Test-Path -LiteralPath $InstallerOutput -PathType Leaf)) {
            throw "Inno Setup completed without the expected installer: $InstallerOutput"
        }
        if ((Get-Item -LiteralPath $InstallerOutput).Length -le 0) {
            throw "Inno Setup produced an empty installer: $InstallerOutput"
        }
    }
    else {
        Write-Host "Inno Setup 6 was not found; the ZIP bundle is the complete distributable output."
    }
}
else {
    Write-Host "Optional Inno Setup compilation was explicitly skipped."
}

Assert-GitReleaseStateUnchanged -InitialState $InitialGitState -GitCommand $GitCommand

Write-Host ""
Write-Host "Windows release build completed successfully:"
Write-Host "  ZIP bundle: $BundleZip"
Write-Host "  ZIP SHA-256: $BundleZipSha"
if ($null -ne $InstallerOutput) {
    Write-Host "  Installer:  $InstallerOutput"
}
Write-Host "  Collector:  $CollectorExe"
Write-Host "  Data Studio: $DataStudioExe"
