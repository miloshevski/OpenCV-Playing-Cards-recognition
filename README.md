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
python read_indices_from_out_cards_v2.py \
  --roi_w 0.18 \
  --roi_h 0.312 \
  --split_y 0.568 \
  --left_band 0.48 \
  --max_rel_area 0.15 \
  --min_area 19 \
  --debug
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

isechi gi slikite python seci.py

prepoznaj python match_all.py
