# 🃏 Детекција и читање на карти

Овој процес се состои од **5 чекори**:

---

## 1️⃣ Детектирање на карти

```bash
python cards_otsu_detect.py test.jpg --warp-dir out_cards --mask-out mask.png --panel-out panel.jpg
```

➤ Ги детектира сите карти на слика, ги запишува како поединечни слики во `out_cards/`  
➤ Дополнително, генерира маска и прегледен панел

---

## 2️⃣ Читање на индекс (ранг + боја)

```bash
python read_indices_from_out_cards_v2.py   --roi_w 0.18   --roi_h 0.312   --split_y 0.568   --left_band 0.48   --max_rel_area 0.15   --min_area 19   --debug
```

➤ Ги чита исечените карти од `out_cards/`  
➤ Автоматски детектира `rank` и `suit`  
➤ Запишува резултати во `out_cards_indices.json`  
➤ Генерира дебаг слики во `_debug_indices/`

---

## 3️⃣ Исекување на binarized `rank` и `suit` региони

```bash
python extract_crops_and_predict.py
```

➤ Ги вади binarized деловите за `rank` и `suit`  
➤ Ги зачувува во `crops/`  
➤ Прикажува предикција за секоја карта

---

## 4️⃣ Сечење и центрирање на `crops`

```bash
python seci.py
```

➤ Ги обработува `crops/` и ги форматира за натамошна анализа

---

## 5️⃣ Финално препознавање

```bash
python match_all.py
```

➤ Ги чита `crops` и споредува со темплејти од `templates/ranks` и `templates/suits`  
➤ Го создава `final_cards.json` со детектирани карти

---

## ✅ Целосно автоматизирано

```bash
python main.py test.jpg
```

➤ Ги извршува сите чекори погоре последователно  
➤ На крај враќа резултат: `["7c", "As", "10d"]`

---

## 🚀 FastAPI endpoint

```bash
uvicorn api:app --reload
```

➤ Достапно на `http://127.0.0.1:8000/predict`  
➤ Поддржува `POST` слика, враќа JSON со карти

---

## 📌 Инструкции за прилагодување и користење

За да можете успешно да ја прилагодите оваа програма за вашите потреби, потребно е да користите **слики од вашите сопствени карти** за попрецизно препознавање.

- Во `templates/ranks/` ставете исечени слики од бројките на сите карти (од A до K), **бинаризирани** и во **резолуција 70x120** пиксели.
- Во `templates/suits/` ставете ги симболите (срце, детелина, лист, клопче) **бинаризирани** во **70x70** пиксели.

🔹 Откако ќе ги внесете вашите слики, стартувајте еднаш:

```bash
python seci.py
```

🔹 Потоа, сликајте ваши карти на **црна позадина** и зачувајте ја сликата (на пример `test.jpg`). Извршете:

```bash
python cards_otsu_detect.py test.jpg --warp-dir out_cards --mask-out mask.png --panel-out panel.jpg
```

🔹 Ќе добиете `out_cards/` со исечени карти од сликата. За калибрација извршете:

```bash
python measure_index_coords.py
```

Во прозорецот што ќе се отвори:

1. Обележете правоаголник околу бројката (Enter)
2. Обележете правоаголник околу симболот (Enter)
3. Притиснете Esc

Во `_measure/card_01_measure.json` ќе се појави `suggested_params`. Користете ги тие параметри при:

```bash
python read_indices_from_out_cards_v2.py --roi_w ... --roi_h ... итн.
```

🔹 Потоа:

```bash
python extract_crops_and_predict.py
python match_all.py
```

🔹 Или автоматски:

```bash
python main.py test.jpg
```

---

## 🌐 FastAPI

За стартување на API серверот:

```bash
uvicorn api:app --reload
```

➤ Потоа можете да испраќате слики преку `POST /predict` и ќе добиете JSON со сите детектирани карти.
