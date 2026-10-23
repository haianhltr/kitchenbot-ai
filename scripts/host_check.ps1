<#
.SYNOPSIS
    KitchenBot Windows host check: is this machine ready for the Isaac Lab 3.0 / Isaac Sim 6.2 install (issue #1)?

.DESCRIPTION
    Check mode (default) is read-only. It prints one line per check,

        [PASS] Long paths: LongPathsEnabled=1
        [FAIL] Long paths: LongPathsEnabled=0 -> run as admin: scripts\host_check.ps1 -Fix, then reboot

    then a summary line, and exits with code 1 if any check is FAIL (0 otherwise). WARN never changes the
    exit code. Results are written with Write-Output so they can be captured.

    -Fix applies the three fixes this script is allowed to make, then runs the checks:
      1. LongPathsEnabled=1 in HKLM (needs an elevated shell; if not elevated it prints the exact command),
      2. git config --global core.longpaths true,
      3. installs uv with the official installer if it is missing.
    It never installs Visual Studio or an NVIDIA driver; the FAIL lines carry the download links.

    Windows PowerShell 5.1 compatible (no &&, no ternary, no ??). ASCII only, so 5.1 reads it correctly.

.PARAMETER Fix
    Apply the fixes listed above before checking. Run it from an elevated PowerShell for the long paths fix.

.PARAMETER VenvRoot
    Folder that will hold the virtual environments (docs/setup.md section 3). Only its drive is used,
    for the free-space check. Nothing is created there.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\host_check.ps1

.EXAMPLE
    # In an elevated PowerShell:
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\host_check.ps1 -Fix
#>
[CmdletBinding()]
param(
    [switch]$Fix,
    [string]$VenvRoot = 'C:\venvs'
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------------------------------------

# Minimum NVIDIA driver for the CUDA 13.0 PyTorch wheels (torch 2.11.0 cu130). NVIDIA's table "CUDA Toolkit
# and Corresponding Driver Versions" lists CUDA 13.0 as Linux >=580.65.06 (GA) / >=580.95.05 (Update 2) and
# gives "N/A" for Windows, because the Windows driver is no longer bundled from CUDA 13.0 on. The Windows
# number is therefore the R580 driver branch ("CUDA Toolkit and Corresponding Driver Branch": CUDA 13.0 ->
# R580) and the minor-version-compatibility row "13.x: driver >= 580". Compared numerically as major.minor.
#   https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
#   https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html  (Table 3, 13.0 rows)
# Read 2026-10-23.
$MinDriverCuda130 = [version]'580.0'

# The Windows driver NVIDIA tested Isaac Sim 6.2 with. Older but >= 580 still passes the CUDA check, so this
# is a WARN, not a FAIL. https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html
$TestedDriverIsaacSim = [version]'595.97'

$FailFreeGB  = 40     # issue #35: FAIL under 40 GB free (Isaac Sim itself lists 50 GB minimum)
$WarnFreeGB  = 80     # WARN under 80 GB free (docs/setup.md budgets 100 GB)
$MinRamGiB   = 31     # Isaac Sim lists 32 GB. A 32 GB machine reports about 31.6 GiB, so 31 avoids a false WARN.
$MinVramMiB  = 15000  # Isaac Sim lists 16 GB VRAM. A 16 GB card reports about 16300 MiB in nvidia-smi.

$LongPathsKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem'
$LongPathsFixCommand = "New-ItemProperty -Path '$LongPathsKey' -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force"
$UvInstallCommand = 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'

$VsBuildToolsUrl = 'https://aka.ms/vs/17/release/vs_BuildTools.exe'
$GitUrl          = 'https://git-scm.com/download/win'
$NvidiaDriverUrl = 'https://www.nvidia.com/Download/index.aspx'
$Python312Url    = 'https://www.python.org/downloads/windows/'

$script:Counts = @{ PASS = 0; WARN = 0; FAIL = 0 }
$script:FixedLongPaths = $false

# ---------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------

function Add-Result {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('PASS', 'WARN', 'FAIL')][string]$Status,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Detail,
        [string]$Advice = ''
    )
    $script:Counts[$Status] = $script:Counts[$Status] + 1
    $line = '[{0}] {1}: {2}' -f $Status, $Name, $Detail
    if ($Status -ne 'PASS' -and $Advice) {
        $line = $line + ' -> ' + $Advice
    }
    Write-Output $line
}

function Format-Num {
    param([double]$Value, [string]$Format = '0.0')
    return $Value.ToString($Format, [System.Globalization.CultureInfo]::InvariantCulture)
}

# Path of an executable on PATH, or $null. Application only, so aliases and functions do not match.
function Find-Tool {
    param([string]$Name)
    $cmd = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { return $cmd.Source }
    return $null
}

# Run an executable and capture stdout, stderr and the exit code without PowerShell's native-stderr quirks.
function Invoke-Tool {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [int]$TimeoutMs = 30000
    )
    $quoted = @()
    foreach ($a in $Arguments) {
        if ($a -match '\s') { $quoted += ('"' + $a + '"') } else { $quoted += $a }
    }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FilePath
    $psi.Arguments = ($quoted -join ' ')
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $proc = [System.Diagnostics.Process]::Start($psi)
    $outTask = $proc.StandardOutput.ReadToEndAsync()
    $errTask = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit($TimeoutMs)) {
        try { $proc.Kill() } catch { }
        return [pscustomobject]@{ ExitCode = -1; Out = ''; Err = 'timed out' }
    }
    $proc.WaitForExit()
    return [pscustomobject]@{ ExitCode = $proc.ExitCode; Out = $outTask.Result.Trim(); Err = $errTask.Result.Trim() }
}

function Test-IsAdmin {
    return ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-UvPath {
    $onPath = Find-Tool 'uv'
    if ($onPath) { return $onPath }
    $candidates = @(
        (Join-Path $env:USERPROFILE '.local\bin\uv.exe'),
        (Join-Path $env:USERPROFILE '.cargo\bin\uv.exe')
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

# Runs one check; an unexpected exception becomes a FAIL line instead of a crash.
function Invoke-Check {
    param([string]$Name, [scriptblock]$Body)
    try {
        & $Body
    } catch {
        Add-Result 'FAIL' $Name ('check crashed: ' + $_.Exception.Message) 'report this output to kb-sim'
    }
}

# ---------------------------------------------------------------------------------------------------------
# -Fix actions (never touch Visual Studio or the NVIDIA driver)
# ---------------------------------------------------------------------------------------------------------

function Invoke-Fixes {
    Write-Output '--- -Fix: applying fixes ---'

    # 1. Long paths (HKLM, needs an elevated shell)
    $current = $null
    try { $current = (Get-ItemProperty -Path $LongPathsKey -Name LongPathsEnabled -ErrorAction Stop).LongPathsEnabled } catch { $current = $null }
    if ($current -eq 1) {
        Write-Output '[SKIP] Long paths: LongPathsEnabled is already 1'
    } elseif (-not (Test-IsAdmin)) {
        Write-Output ('[SKIP] Long paths: this shell is not elevated -> open PowerShell as Administrator and run: ' + $LongPathsFixCommand)
        Write-Output '       (or run scripts\host_check.ps1 -Fix from that elevated shell), then reboot'
    } else {
        try {
            New-ItemProperty -Path $LongPathsKey -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force | Out-Null
            $script:FixedLongPaths = $true
            Write-Output '[FIXED] Long paths: set LongPathsEnabled=1 (reboot required before it takes effect)'
        } catch {
            Write-Output ('[ERROR] Long paths: could not write the registry value: ' + $_.Exception.Message)
        }
    }

    # 2. git core.longpaths
    $git = Find-Tool 'git'
    if (-not $git) {
        Write-Output ('[SKIP] git core.longpaths: git is not installed -> install Git for Windows (' + $GitUrl + ') and re-run -Fix')
    } else {
        try {
            $r = Invoke-Tool $git @('config', '--global', '--bool', '--get', 'core.longpaths')
            if ($r.ExitCode -eq 0 -and $r.Out -eq 'true') {
                Write-Output '[SKIP] git core.longpaths: already true'
            } else {
                $s = Invoke-Tool $git @('config', '--global', 'core.longpaths', 'true')
                if ($s.ExitCode -eq 0) {
                    Write-Output '[FIXED] git core.longpaths: set to true (git config --global)'
                } else {
                    Write-Output ('[ERROR] git core.longpaths: git config failed: ' + $s.Err)
                }
            }
        } catch {
            Write-Output ('[ERROR] git core.longpaths: ' + $_.Exception.Message)
        }
    }

    # 3. uv, official installer
    $uv = Get-UvPath
    if ($uv) {
        Write-Output ('[SKIP] uv: already installed at ' + $uv)
    } else {
        try {
            Write-Output ('[RUN] uv: ' + $UvInstallCommand)
            $ps = Find-Tool 'powershell'
            if (-not $ps) { $ps = 'powershell.exe' }
            $r = Invoke-Tool $ps @('-NoProfile', '-ExecutionPolicy', 'ByPass', '-Command', 'irm https://astral.sh/uv/install.ps1 | iex') 300000
            if ($r.ExitCode -eq 0) {
                # The installer edits the user PATH; make uv visible to the checks below in this session.
                $env:Path = (Join-Path $env:USERPROFILE '.local\bin') + ';' + (Join-Path $env:USERPROFILE '.cargo\bin') + ';' + $env:Path
                Write-Output '[FIXED] uv: installed with the official installer (open a new shell so PATH picks it up)'
            } else {
                $why = $r.Err
                if (-not $why) { $why = $r.Out }
                Write-Output ('[ERROR] uv: installer exited with code ' + $r.ExitCode + ': ' + $why)
            }
        } catch {
            Write-Output ('[ERROR] uv: ' + $_.Exception.Message)
        }
    }

    Write-Output '[INFO] -Fix never installs Visual Studio or an NVIDIA driver; the download links are in the FAIL lines below'
    Write-Output '--- checks ---'
}

# ---------------------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------------------

function Test-WindowsBuild {
    $os = Get-CimInstance Win32_OperatingSystem
    $build = [int]$os.BuildNumber
    $is64 = [Environment]::Is64BitOperatingSystem
    $arch = 'x86'
    if ($is64) { $arch = '64-bit' }
    $caption = ($os.Caption -replace '^Microsoft\s+', '')
    $display = ''
    $ubr = ''
    try {
        $cv = Get-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -ErrorAction Stop
        if ($cv.DisplayVersion) { $display = ' ' + $cv.DisplayVersion }
        if ($cv.UBR) { $ubr = '.' + $cv.UBR }
    } catch { }
    $detail = '{0}{1}, build {2}{3}, {4}' -f $caption, $display, $build, $ubr, $arch
    # Windows 11 is build 22000 or newer. (The registry ProductName still says "Windows 10" on Windows 11.)
    if ($is64 -and $build -ge 22000) {
        Add-Result 'PASS' 'Windows' $detail
    } else {
        Add-Result 'FAIL' 'Windows' $detail 'Isaac Sim 6.2 needs 64-bit Windows 11 (Windows 10 is not supported); upgrade the OS or use the Ubuntu 24.04 fallback in docs/setup.md section 5'
    }
}

function Test-LongPathsRegistry {
    $v = $null
    try { $v = (Get-ItemProperty -Path $LongPathsKey -Name LongPathsEnabled -ErrorAction Stop).LongPathsEnabled } catch { $v = $null }
    if ($v -eq 1) {
        Add-Result 'PASS' 'Long paths' 'LongPathsEnabled=1'
    } else {
        $shown = 'not set'
        if ($null -ne $v) { $shown = [string]$v }
        $advice = 'run as admin: scripts\host_check.ps1 -Fix, then reboot'
        Add-Result 'FAIL' 'Long paths' ('LongPathsEnabled=' + $shown) $advice
    }
}

function Test-GitLongPaths {
    $git = Find-Tool 'git'
    if (-not $git) {
        Add-Result 'FAIL' 'git core.longpaths' 'git not found on PATH' ('install Git for Windows: ' + $GitUrl + ', then run scripts\host_check.ps1 -Fix')
        return
    }
    $r = Invoke-Tool $git @('config', '--global', '--bool', '--get', 'core.longpaths')
    if ($r.ExitCode -eq 0 -and $r.Out -eq 'true') {
        Add-Result 'PASS' 'git core.longpaths' 'core.longpaths=true'
    } else {
        $shown = 'not set'
        if ($r.ExitCode -eq 0 -and $r.Out) { $shown = $r.Out }
        Add-Result 'FAIL' 'git core.longpaths' ('core.longpaths=' + $shown) 'run: git config --global core.longpaths true (or scripts\host_check.ps1 -Fix)'
    }
}

# vswhere -format json -> list of instances. Windows PowerShell 5.1 emits the whole JSON array as ONE object
# (an empty "[]" counts as 1 under @(...)), so flatten it explicitly.
function Get-VsInstances {
    param([string]$Json)
    $instances = New-Object System.Collections.ArrayList
    if (-not $Json) { return ,$instances }
    $parsed = ConvertFrom-Json -InputObject $Json
    foreach ($outer in $parsed) {
        foreach ($item in $outer) {
            if ($null -ne $item -and $item.PSObject.Properties['installationVersion']) { [void]$instances.Add($item) }
        }
    }
    return ,$instances
}

function Test-BuildTools {
    $name = 'VS 2022 Build Tools (C++)'
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $advice = 'install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload: ' + $VsBuildToolsUrl + ' (no reboot needed)'
    if (-not (Test-Path $vswhere)) {
        Add-Result 'FAIL' $name 'vswhere.exe not found, so no Visual Studio is installed' $advice
        return
    }
    $component = 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
    $r = Invoke-Tool $vswhere @('-products', '*', '-requires', $component, '-version', '[17.0,18.0)', '-format', 'json')
    $found = Get-VsInstances ''
    if ($r.ExitCode -eq 0) { $found = Get-VsInstances $r.Out }
    if ($found.Count -gt 0) {
        $first = $found[0]
        Add-Result 'PASS' $name ('{0} {1} with the C++ x64 toolset' -f $first.displayName, $first.installationVersion)
        return
    }
    # No VS 2022 with the C++ toolset. Say what is there instead of just "missing".
    $any = Invoke-Tool $vswhere @('-products', '*', '-requires', $component, '-format', 'json')
    $other = Get-VsInstances ''
    if ($any.ExitCode -eq 0) { $other = Get-VsInstances $any.Out }
    if ($other.Count -gt 0) {
        $o = $other[0]
        Add-Result 'WARN' $name ('{0} {1} has the C++ toolset, but it is not VS 2022 (17.x)' -f $o.displayName, $o.installationVersion) ('the docs target VS 2022; install it: ' + $VsBuildToolsUrl)
        return
    }
    Add-Result 'FAIL' $name 'Visual Studio found, but no VC++ x64 toolset (Microsoft.VisualStudio.Component.VC.Tools.x86.x64)' $advice
}

function Test-Uv {
    $uv = Get-UvPath
    $advice = 'run scripts\host_check.ps1 -Fix, or: ' + $UvInstallCommand
    if (-not $uv) {
        Add-Result 'FAIL' 'uv' 'uv not found' $advice
        return
    }
    $r = Invoke-Tool $uv @('--version')
    if ($r.ExitCode -ne 0) {
        Add-Result 'FAIL' 'uv' ('uv --version failed (exit {0}): {1}' -f $r.ExitCode, $r.Err) $advice
        return
    }
    if (Find-Tool 'uv') {
        Add-Result 'PASS' 'uv' $r.Out
    } else {
        Add-Result 'WARN' 'uv' ($r.Out + ' is installed at ' + $uv + ' but is not on PATH') 'open a new PowerShell window so the installer PATH change applies'
    }
}

function Test-Python312 {
    $py = Find-Tool 'py'
    $advice = 'optional, because uv downloads Python 3.12 itself on "uv venv --python 3.12"; to have it anyway install from ' + $Python312Url + ' or run: uv python install 3.12'
    if (-not $py) {
        Add-Result 'WARN' 'Python 3.12' 'py launcher not found' $advice
        return
    }
    $r = Invoke-Tool $py @('-3.12', '--version')
    if ($r.ExitCode -eq 0 -and $r.Out -match 'Python 3\.12') {
        Add-Result 'PASS' 'Python 3.12' $r.Out
    } else {
        Add-Result 'WARN' 'Python 3.12' 'py -3.12 found no Python 3.12' $advice
    }
}

function Test-NvidiaGpu {
    $smi = Find-Tool 'nvidia-smi'
    if (-not $smi) {
        $smiDefault = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'
        if (Test-Path $smiDefault) { $smi = $smiDefault }
    }
    $driverAdvice = 'install a current NVIDIA Game Ready or Studio driver: ' + $NvidiaDriverUrl
    if (-not $smi) {
        Add-Result 'FAIL' 'NVIDIA GPU' 'nvidia-smi not found, so no NVIDIA driver is installed' $driverAdvice
        Add-Result 'FAIL' 'NVIDIA driver' 'unknown (no nvidia-smi)' $driverAdvice
        return
    }
    $r = Invoke-Tool $smi @('--query-gpu=name,driver_version,memory.total', '--format=csv,noheader')
    if ($r.ExitCode -ne 0 -or -not $r.Out) {
        $why = $r.Err
        if (-not $why) { $why = $r.Out }
        Add-Result 'FAIL' 'NVIDIA GPU' ('nvidia-smi failed (exit {0}): {1}' -f $r.ExitCode, $why) 'reinstall the NVIDIA driver, reboot, and re-run'
        Add-Result 'FAIL' 'NVIDIA driver' 'unknown (nvidia-smi failed)' $driverAdvice
        return
    }
    $lines = @($r.Out -split "`r?`n" | Where-Object { $_.Trim() -ne '' })
    $first = $lines[0]
    $parts = @($first -split ',' | ForEach-Object { $_.Trim() })
    if ($parts.Count -lt 3) {
        Add-Result 'FAIL' 'NVIDIA GPU' ('unexpected nvidia-smi output: ' + $first) 'run nvidia-smi by hand and report its output'
        return
    }
    $gpuName = $parts[0]
    $driverText = $parts[1]
    $vramMiB = 0.0
    if ($parts[2] -match '(\d+)') { $vramMiB = [double]$Matches[1] }
    $vramGB = $vramMiB / 1024.0

    # GPU name and VRAM
    $gpuDetail = '{0}, {1} MiB ({2} GB VRAM)' -f $gpuName, [int]$vramMiB, (Format-Num $vramGB)
    if ($lines.Count -gt 1) { $gpuDetail = $gpuDetail + ', ' + $lines.Count + ' GPUs listed, first one shown' }
    if ($vramMiB -ge $MinVramMiB) {
        Add-Result 'PASS' 'NVIDIA GPU' $gpuDetail
    } else {
        Add-Result 'WARN' 'NVIDIA GPU' $gpuDetail 'Isaac Sim lists 16 GB VRAM as the minimum; use fewer envs and expect out-of-memory on the kitchen scene'
    }

    # Driver version, compared numerically as major.minor
    $driver = $null
    try { $driver = [version]$driverText } catch { $driver = $null }
    if ($null -eq $driver) {
        Add-Result 'FAIL' 'NVIDIA driver' ('could not parse driver version "' + $driverText + '"') 'run nvidia-smi by hand and report its output'
        return
    }
    $limits = 'min {0} for CUDA 13.0, Isaac Sim tested {1}' -f (Format-Num $MinDriverCuda130.Major '0'), $TestedDriverIsaacSim.ToString()
    if ($driver -lt $MinDriverCuda130) {
        Add-Result 'FAIL' 'NVIDIA driver' ('{0} ({1})' -f $driverText, $limits) ('update the driver to ' + $TestedDriverIsaacSim.ToString() + ' or newer (CUDA 13.0 needs R580+): ' + $NvidiaDriverUrl)
    } elseif ($driver -lt $TestedDriverIsaacSim) {
        Add-Result 'WARN' 'NVIDIA driver' ('{0} ({1})' -f $driverText, $limits) ('works for CUDA 13.0 but is older than the tested ' + $TestedDriverIsaacSim.ToString() + '; update if Isaac Sim fails to start: ' + $NvidiaDriverUrl)
    } else {
        Add-Result 'PASS' 'NVIDIA driver' ('{0} ({1})' -f $driverText, $limits)
    }
}

function Test-DiskSpace {
    $drives = @('C:')
    $venvDrive = [System.IO.Path]::GetPathRoot($VenvRoot).TrimEnd('\').ToUpper()
    if ($venvDrive -and ($drives -notcontains $venvDrive)) { $drives += $venvDrive }
    foreach ($d in $drives) {
        $name = 'Disk space'
        $role = ''
        if ($d -eq $venvDrive) { $role = ' (holds ' + $VenvRoot + ')' }
        $disk = Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='{0}'" -f $d)
        if (-not $disk) {
            Add-Result 'FAIL' $name ($d + ' drive not found' + $role) 'point -VenvRoot at a drive that exists, or free up C:'
            continue
        }
        $freeGB = [double]$disk.FreeSpace / 1GB
        $sizeGB = [double]$disk.Size / 1GB
        $detail = '{0} {1} GB free of {2} GB{3}' -f $d, (Format-Num $freeGB), (Format-Num $sizeGB), $role
        if ($freeGB -lt $FailFreeGB) {
            Add-Result 'FAIL' $name $detail ('free space on {0} (the install alone is 30 GB or more; Isaac Sim lists 50 GB, docs/setup.md budgets 100 GB)' -f $d)
        } elseif ($freeGB -lt $WarnFreeGB) {
            Add-Result 'WARN' $name $detail ('docs/setup.md budgets 100 GB; free at least {0} GB on {1} before long training runs' -f $WarnFreeGB, $d)
        } else {
            Add-Result 'PASS' $name $detail
        }
    }
}

function Test-PhysicalRam {
    $bytes = [double](Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    $gib = $bytes / 1GB
    $detail = '{0} GB installed' -f (Format-Num $gib)
    if ($gib -lt $MinRamGiB) {
        Add-Result 'WARN' 'RAM' $detail 'Isaac Sim lists 32 GB as the minimum; close other programs and lower num_envs'
    } else {
        Add-Result 'PASS' 'RAM' $detail
    }
}

function Test-RebootPending {
    $reasons = @()
    if (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending') {
        $reasons += 'Component Based Servicing\RebootPending'
    }
    if (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired') {
        $reasons += 'WindowsUpdate\Auto Update\RebootRequired'
    }
    $pfro = Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager' -Name PendingFileRenameOperations -ErrorAction SilentlyContinue
    if ($pfro -and $pfro.PendingFileRenameOperations) {
        $n = @($pfro.PendingFileRenameOperations | Where-Object { $_ }).Count
        $reasons += ('Session Manager\PendingFileRenameOperations ({0} entries)' -f $n)
    }
    if ($reasons.Count -eq 0) {
        Add-Result 'PASS' 'Reboot pending' 'no'
    } else {
        Add-Result 'WARN' 'Reboot pending' ('yes: ' + ($reasons -join '; ')) 'reboot once before the install (the long paths change needs a reboot anyway); PendingFileRenameOperations alone is often harmless'
    }
}

# ---------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------

$mode = 'check mode, read-only'
if ($Fix) { $mode = '-Fix mode' }
Write-Output ('KitchenBot host check ({0}) on {1}, Windows PowerShell {2}' -f $mode, $env:COMPUTERNAME, $PSVersionTable.PSVersion.ToString())

if ($Fix) { Invoke-Fixes }

Invoke-Check 'Windows' { Test-WindowsBuild }
Invoke-Check 'Long paths' { Test-LongPathsRegistry }
Invoke-Check 'git core.longpaths' { Test-GitLongPaths }
Invoke-Check 'VS 2022 Build Tools (C++)' { Test-BuildTools }
Invoke-Check 'uv' { Test-Uv }
Invoke-Check 'Python 3.12' { Test-Python312 }
Invoke-Check 'NVIDIA GPU' { Test-NvidiaGpu }
Invoke-Check 'Disk' { Test-DiskSpace }
Invoke-Check 'RAM' { Test-PhysicalRam }
Invoke-Check 'Reboot pending' { Test-RebootPending }

$pass = $script:Counts['PASS']
$warn = $script:Counts['WARN']
$fail = $script:Counts['FAIL']
if ($fail -gt 0) {
    Write-Output ('Summary: {0} PASS, {1} WARN, {2} FAIL -> NOT READY: fix the FAIL lines above, then re-run scripts\host_check.ps1' -f $pass, $warn, $fail)
} elseif ($warn -gt 0) {
    Write-Output ('Summary: {0} PASS, {1} WARN, {2} FAIL -> READY, read the WARN lines first' -f $pass, $warn, $fail)
} else {
    Write-Output ('Summary: {0} PASS, {1} WARN, {2} FAIL -> READY' -f $pass, $warn, $fail)
}
if ($script:FixedLongPaths) {
    Write-Output 'Note: reboot now so the long paths setting takes effect, then re-run this script without -Fix.'
}

if ($fail -gt 0) { exit 1 }
exit 0
