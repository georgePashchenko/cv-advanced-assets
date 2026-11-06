# CV Advanced · проміжні артефакти

Зображення й графіки для слайдів курсу «CV для Deftech та автономних систем», а також скрипти, якими їх згенеровано.
Репозиторій публічний: Google Slides вставляє картинки лише за публічним URL.

## Як вставити файл у слайд

Пряме посилання на файл:

```
https://raw.githubusercontent.com/georgePashchenko/cv-advanced-assets/main/<шлях до файлу>
```

Наприклад, `lesson11/slide13/p13_fmcw_chart.png`. Slides копіює зображення в момент вставки. Тому після оновлення файлу
картинку на слайді треба замінити ще раз (`replaceImage`), бо сама вона не оновиться.

## Структура

```
lessonNN/slideMM/            фінальні версії для слайда MM заняття NN
lessonNN/slideMM/versions/   оригінали та проміжні варіанти
scripts/                     генерація та редагування
```

## Заняття 11

| Слайд | Файл | Що це |
|---|---|---|
| 5 | `p5_camera.png`, `p5_lidar.png`, `p5_radar.png` | Один кадр KITTI object 000008 очима трьох сенсорів: RGB з лупою пікселів; справжні точки HDL-64E, спроєктовані в камеру; радарні детекції (R, θ), **змодельовані** з розмітки, бо в KITTI радара немає |
| 10 | `p10_meme.jpg` | Мем «Max range 200 м» проти «40 м @ 10 % відбивної здатності» на оригінальному шаблоні |
| 10 | `p10_infographic.jpg` | Інфографіка «П'ять архітектур розгортання променів у LiDAR» із блоком «Аналіз datasheet» |
| 10 | `p10_scan_*.png` | Шаблони сканування (азимут × кут місця), колір = час точки в кадрі 10 Гц |
| 13 | `p13_fmcw_chart.png` | FMCW: TX/RX-chirp, τ і f_b, range FFT з піками 30 м і 100 м |

## Скрипти

Потрібен Python із `numpy`, `matplotlib` і `Pillow`.

| Скрипт | Призначення |
|---|---|
| `gemini_edit.py` | Редагування зображення через Gemini API: ключ береться з `GEMINI_API_KEY`, розмір задається через `GEMINI_IMAGE_SIZE` |
| `meme_text.py` | Підписи на шаблоні мему |
| `p10_scan_patterns.py` | Діаграми шаблонів сканування LiDAR |
| `relayout.py` | Перекомпонування інфографіки слайда 10 з перевірених фрагментів |
| `patch_text.py` | Точкова заміна тексту на растрі |
| `p13_fmcw_chart.py` | Графік FMCW для слайда 13 |
| `p5_sensor_outputs.py` | Камера, LiDAR і радар на одному кадрі KITTI для слайда 5 (дані: `python tools/fetch_kitti.py --object` у репозиторії курсу) |
| `drive_id.py` | ID файлів Google Drive з локальної бази Drive for Desktop |

Gemini добре дає оформлення, але плутає текст і координати. Тому кожен результат потрібно вичитувати. Графіки з точними
даними будуються в matplotlib, а Gemini слугує лише для макета.

## Ліцензії даних

Кадри та скани KITTI: Geiger et al., *Vision meets Robotics: The KITTI Dataset*, IJRR 2013 ·
[cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti/). Ліцензія CC BY-NC-SA 3.0, тобто лише некомерційне використання.

Радарні детекції для слайда 18 (`p18_egomotion.png`): Schumann et al., *RadarScenes: A Real-World Radar Point Cloud
Data Set for Automotive Applications*, FUSION 2021 · [zenodo.org/record/4559821](https://zenodo.org/record/4559821),
sequence_90. Ліцензія CC BY-NC-SA 4.0. Одну послідовність можна взяти з 11-ГБ архіву без повного завантаження:
`python scripts/http_zip.py <url> get RadarScenes/data/sequence_90/radar_data.h5 seq90.h5`.
