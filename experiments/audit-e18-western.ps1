param(
    [Parameter(Mandatory)][string]$Data,
    [Parameter(Mandatory)][string]$Output
)
$ErrorActionPreference = 'Stop'
$dataPath = (Resolve-Path -LiteralPath $Data).Path
$ingest = Get-Content -Raw -LiteralPath (Join-Path $dataPath 'outputs/ingest.json') | ConvertFrom-Json
$ingested = [System.Collections.Generic.HashSet[string]]::new([string[]]$ingest.input_file_names)
$results = foreach ($year in 2018..2025) {
    $manifest = Join-Path $dataPath "outputs/download_$year-01-01_$($year)-12-31.json"
    $entries = Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json
    foreach ($source in 'MODIS_SP', 'VIIRS_SNPP_SP') {
        $rows = @($entries | Where-Object source -eq $source | Sort-Object start)
        $next = [datetime]::new($year,1,1)
        $bytes = 0L
        foreach ($row in $rows) {
            $start = [datetime]::ParseExact($row.start, 'yyyy-MM-dd', [cultureinfo]::InvariantCulture)
            if ($start -ne $next -or $row.days -le 0) { throw "Gap or overlap in $manifest for $source at $next" }
            $name = "$($source)_-125_32_-102_49_$($row.start)_$($row.days)d.csv"
            $file = Get-Item -LiteralPath (Join-Path $dataPath "raw/$name")
            if ($file.Length -ne $row.bytes -or !$ingested.Contains($name)) {
                throw "File size or ingest list mismatch for $name"
            }
            $bytes += $file.Length
            $next = $start.AddDays($row.days)
        }
        if ($next -ne [datetime]::new($year + 1,1,1)) { throw "Incomplete year $year for $source" }
        [pscustomobject]@{ year = $year; product = $source; slices = $rows.Count; bytes = $bytes;
            status = 'continuous_manifest_and_sizes_match_ingest' }
    }
}
$results | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath $Output

