param(
    [Parameter(Mandatory)][string]$Data,
    [Parameter(Mandatory)][string]$Run,
    [Parameter(Mandatory)][string]$Output
)
$ErrorActionPreference = 'Stop'
$region = 'eastern_us'
$sample = Import-Csv -LiteralPath (Join-Path $Run 'coverage_sample.csv') | Where-Object region -eq $region
$zero = $sample | Where-Object { [long]$_.aqua -eq 0 } | Select-Object -First 1
$positive = $sample | Where-Object { [long]$_.aqua -gt 0 } | Select-Object -First 1
$cache = @{}
foreach ($experiment in 'e12','e13') {
    Get-ChildItem -File -Recurse -LiteralPath (Join-Path $Data "outputs/$experiment/swaths") |
        ForEach-Object { $cache[$_.BaseName] = $_.Length }
}
$collections = [ordered]@{ MYD14 = 'C2278858993-LPCLOUD'; MYD03 = 'C1379841358-LAADS';
    VNP14IMG = 'C2734202914-LPCLOUD'; VNP03IMG = 'C2105092163-LAADS' }
$assets = foreach ($day in @($zero, $positive)) {
    $start = [datetime]::ParseExact($day.date_utc, 'yyyy-MM-dd', [cultureinfo]::InvariantCulture)
    $range = $start.ToString('yyyy-MM-ddTHH:mm:ssZ') + ',' + $start.AddDays(1).ToString('yyyy-MM-ddTHH:mm:ssZ')
    foreach ($product in $collections.Keys) {
        $uri = "https://cmr.earthdata.nasa.gov/search/granules.json?collection_concept_id=$($collections[$product])&bounding_box=-88,32,-76,43&temporal=$range&page_size=500"
        $response = Invoke-WebRequest -Uri $uri -TimeoutSec 35
        $entries = @(($response.Content | ConvertFrom-Json).feed.entry)
        if ($entries.Count -ne [int]@($response.Headers['CMR-Hits'])[0]) { throw "Incomplete CMR query $uri" }
        foreach ($asset in $entries) {
            $id = $asset.producer_granule_id -replace '\.(hdf|nc)$',''
            [pscustomobject]@{
                region = $region; date_utc = $day.date_utc; aqua = $day.aqua; snpp = $day.snpp
                product = $product; granule = $id; catalog_mb = $asset.granule_size
                cached_bytes = $cache[$id]; cached = $cache.ContainsKey($id)
            }
        }
    }
}
$assets | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath $Output

