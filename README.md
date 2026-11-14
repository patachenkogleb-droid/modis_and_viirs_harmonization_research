# Відтворення гармонізації Aqua MODIS → S‑NPP VIIRS

Почніть зі [українського звіту](research-report_uk.pdf) або [англійського звіту](research-report_en.pdf) та [CSV із вісьмома коефіцієнтами](modis_viirs_coefficients.csv). Для кількості хотспотів за повний UTC-місяць фазова модель E19 використовує `k_hotspot_day × Aqua_D + k_hotspot_night × Aqua_N`. Якщо доступна лише сума, `k_hotspots_combined` задає окрему історичну модель E18. Для FRP застосовуйте [регіональну таблицю](frp_modis_viirs_coefficients.csv) за правилами [календаря](docs/calendar-coefficients.md). Три `k_frp_*` у зведеному CSV валідовані для Серрадо, загальний резерв дорівнює 1.

Репозиторій для публікації — [GitHub](https://github.com/patachenkogleb-droid/modis_and_viirs_harmonization_research).

## Добова FRP зіставлених прольотів

`k_frp_daily_matched_day=0.732069094` — округлений множник E5/E6 для добових сум зіставлених прольотів у напрямку Aqua → S-NPP. E7 перевірив його на добових сумах пар. `k_frp_daily_night=1.0` залишає нічну FRP у вихідній шкалі. Дослідницьку нічну оцінку 0.891239620 та обидві вибірки описує [огляд E5–E9](docs/frp-day-night-report.md).

```powershell
uv run --locked python -m experiments.e5_scan_edge_daynight --data $Data --grid-size 3000
uv run --locked python -m experiments.e7_daily_calibration --data $Data --grid-size 3000
```

## Відтворення з агрегатів

Із кореня пакета, у PowerShell. Потрібен стабільний Rust із доступом до залежностей `Cargo.lock`. Команди створюють нові каталоги виходу.

```powershell
cargo test --release --locked --manifest-path accel-rs/Cargo.toml
cargo run --release --locked --manifest-path accel-rs/Cargo.toml -- hotspots-e19 --input data/outputs/e18/freeze --output replay/e19
cargo run --release --locked --manifest-path accel-rs/Cargo.toml -- hotspots-e20t --input data/outputs/e20/full --baseline data/outputs/e18/freeze --output replay/e20t
```

`daily.csv` E18 і `monthly_biome_counts.csv` E20 містять агреговані лічильники, необхідні для цих повторів. Очікувані значення містяться у `decision.csv` та `fold_coefficients.csv` відповідних прогонів. Зведений CSV — зафіксований вхідний артефакт. E18 і E19 експортують окремі CSV, E10 — регіональну таблицю FRP. Для збірки зведеного рядка беріть `k_hotspots_combined` з E18, `k_hotspot_day/k_hotspot_night` з E19 і рядки `cerrado,all/D/N` з календарної таблиці FRP.

## Вхідні дані для повного прогону

Використовуйте окремий робочий каталог даних. У командах нижче `$Data = 'data'`; можна задати інший абсолютний шлях. Включено нормалізований Parquet, оригінальні регіональні ZIP для аудиту E9, кліпи земного покриву та країнові CSV, які читає Rust. Для повтору скопіюйте каталог data в окрему робочу папку і задайте її шлях як $Data.

| Вхід | Джерело й розміщення |
|---|---|
| Захід США | [NASA FIRMS Area API](https://firms.modaps.eosdis.nasa.gov/api/area/), стандартні `MODIS_SP` та `VIIRS_SNPP_SP`, bbox `-125,32,-102,49`. Період 2012-01-20–2025-12-31. П'ятиденні CSV у `$Data/raw`, маніфести у `$Data/outputs/download_*.json`. Результат конвертації `$Data/parquet/firms.parquet`. |
| Схід США, Південна Амазонія, Серрадо | [NASA FIRMS Archive Download](https://firms.modaps.eosdis.nasa.gov/download/), стандартні Aqua MODIS C6.1 та S‑NPP VIIRS 375 м C2, 2012-01-20–2020-12-31. По одному набору CSV/ZIP на регіон у `$Data/archive_download/<region>_2012-01-20_2020-12-31`; bbox відповідно `-88,32,-76,43`, `-65,-13,-51,-3`, `-51,-22,-43,-9`. |
| Бореальна Канада, Карпати Румунії, Суматра, Замбія, захід Австралії | [Річні країнові CSV FIRMS](https://firms.modaps.eosdis.nasa.gov/country/). Точні історичні URL і розміри у `data/outputs/e14/download_audit.csv`. Канада, Румунія та Індонезія за 2018–2020, Замбія й Австралія за 2020. Шлях `$Data/outputs/e14/raw/<region>_<year>_<product>.csv`, продукти `modis` та `viirs-snpp`. Rust застосує рамки з `accel-rs/src/hotspots/input.rs`. |
| Земний покрив | Річна мозаїка [OpenLandMap, Zenodo 8338928](https://zenodo.org/records/8338928), [MCD12Q1 v061](https://doi.org/10.5067/MODIS/MCD12Q1.061), шар `LC_Type1`. E9 використовує роки 2012–2020, E20 — 2018–2020 і карту 2020 для заходу США після 2020. Точний інвентар E20 у `data/outputs/e20/full/rasters.csv`. Західні кліпи `$Data/landcover/LC_Type1_<year>.tif`, решта регіонів E9 у `$Data/landcover/<region>/`, країнові кліпи у `$Data/outputs/e14/landcover/<region>/`. |

Номери оригінальних ZIP-експортів FIRMS (MODIS / VIIRS): схід США 841729 / 841730, Південна Амазонія 841731 / 841732, Серрадо 841733 / 841734. Аудит прикордонної колонки E9 зафіксував власника `southern_amazon` для колонки −1641. Відтворення `verify-frp` також використовує країнові входи Австралії 2020 та доступні регіональні архіви. Калібрацію для України в цьому прогоні не перевірено.

Для Area API потрібен власний `FIRMS_MAP_KEY` у локальному `.env` або у файлі, заданому через `FIRMS_ENV_FILE`. Уникайте перекриття між API та архівними CSV для одного сенсора й періоду. Історичні повторні завантаження можуть мати іншу версію або розмір. Порівнюйте з `e18/freeze/inputs.csv`, західним аудитом, `e9/input_checks.json` та інвентарем FRP. Старі абсолютні шляхи в інвентарях фіксують походження файлів і є довідковими.

```powershell
$Data = 'data'
uv sync --locked
# Завантаження та конвертація нижче потрібні лише для реконструкції входів.
uv run --locked python -m src.ingest.firms_client --data $Data download-years (2012..2025)
uv run --locked python -m src.ingest.firms_client --data $Data convert

$boxes = @{eastern_us='-88,32,-76,43'; southern_amazon='-65,-13,-51,-3'; cerrado='-51,-22,-43,-9'}
foreach ($r in $boxes.Keys) {
    $id = "${r}_2012-01-20_2020-12-31"
    uv run --locked python -m src.ingest.firms_client --data $Data convert-archive --raw "$Data/archive_download/$id" --output "$Data/parquet/regions/$id.parquet" "--bbox=$($boxes[$r])"
    uv run --locked python -m src.landcover.download_mosaic "--bbox=$($boxes[$r])" --output-dir "$Data/landcover/$r" (2012..2020)
}
uv run --locked python -m src.landcover.download_mosaic --data $Data (2012..2020)
uv run --locked python -m experiments.e14_country_inputs --output "$Data/outputs/e14"
```

Завантажувач країнових входів E14 запитує 2018–2020 для всіх п'яти країн. Для E18–E20 достатньо 2020 року для Замбії й Австралії. Якщо пакетний запит недоступний, використайте точні URL з інвентарю та розмістіть файли вручну. Оригінальні растри повторно використовуються з відповідних підкаталогів. Для E9 із зовнішнім `$Data` покладіть три включені контрольні CSV E7/E8 у `$Data/outputs` або перегенеруйте їх командами нижче.

```powershell
uv run --locked python -m experiments.e7_daily_calibration --data $Data
uv run --locked python -m experiments.e8_unpaired_detections --data $Data
uv run --locked python -m experiments.e9_regional_daily_calibration --data $Data
uv run --locked python -m experiments.e10_calendar_daily_calibration --data $Data --calendar replay/frp/calendar.csv --output replay/frp/e10

New-Item -ItemType Directory -Force replay | Out-Null
# За наявності сирих західних CSV: ./experiments/audit-e18-western.ps1 -Data $Data -Output replay/western_manifest_audit.csv
cargo run --release --locked --manifest-path accel-rs/Cargo.toml -- hotspots --data $Data --output replay/e18 --western-holdout --freeze
cargo run --release --locked --manifest-path accel-rs/Cargo.toml -- hotspots-e19 --input replay/e18 --output replay/e19-from-raw
cargo run --release --locked --manifest-path accel-rs/Cargo.toml -- hotspots-e20 --data $Data --baseline replay/e18 --output replay/e20
cargo run --release --locked --manifest-path accel-rs/Cargo.toml -- hotspots-e20t --input replay/e20 --baseline replay/e18 --output replay/e20t-from-raw
cargo run --release --locked --manifest-path accel-rs/Cargo.toml -- verify-frp --data $Data --coefficients frp_modis_viirs_coefficients.csv --output replay/frp-verified
```

Для історичної парної моделі E6 використайте `uv run --locked python -m experiments.e6_backtest_validation --data $Data --grid-size 3000`. Параметри фаз і межі регіонів задаються кодом. Каталоги `replay/*` для Rust мають бути відсутні перед запуском. E9 створює `reported_observations.parquet` для E10 і `verify-frp`; він є відновлюваним входом.

## Локальні дані та GitHub

Західний Parquet — основний локальний набір. Список первинних CSV збережено в ingest.json і річних маніфестах FIRMS. Регіональні ZIP потрібні для аудиту E9. Країнові CSV є прямими входами E18/E20 та verify-frp. .gitignore виключає великі вхідні дані зі звичайної публікації коду. Для передачі всього пакета локальні входи потрібно передавати окремо.
