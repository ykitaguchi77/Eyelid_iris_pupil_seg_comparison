$ErrorActionPreference = "Stop"
$Path = if ($args.Length -ge 1) { $args[0] } else { "train_rf-detr.ipynb" }
if (!(Test-Path $Path)) { Write-Error "File not found: $Path" }

$nb = Get-Content -Raw -Encoding UTF8 $Path | ConvertFrom-Json
$cells = @()
foreach ($c in $nb.cells) { if ($c.cell_type -eq 'code') { $cells += $c } }

Write-Output ("Code cells: " + $cells.Count)
$i = 0
foreach ($c in $cells) {
  $i++
  $src = ($c.source -join '')
  $first = if ($src) { ($src -split "`n")[0] } else { "" }
  $exec = if ($c.execution_count) { $c.execution_count } else { "null" }
  $tags = if ($c.metadata -and $c.metadata.tags) { ($c.metadata.tags -join ',') } else { "" }
  Write-Output ("# Cell $i | exec:" + $exec + " | tags:" + $tags)
  if ($env:FULL -eq '1') {
    if ($src) { Write-Output $src }
  } else {
    if ($first) { Write-Output $first }
  }
}
