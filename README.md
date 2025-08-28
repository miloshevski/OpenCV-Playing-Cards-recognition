python cards_otsu_detect.py test2.jpg --warp-dir out_cards --mask-out mask.png --panel-out panel.jpg za detekcija na karte

# 1) осигурај templates/ (ranks/ + suits/), со твои PNG темплејти

# 2) стартувај со debug оверлеи:

python read_indices_from_out_cards.py --in_dir out_cards --templates templates --debug

# ако пак не фаќа, пробај:

python read_indices_from_out_cards.py --idx_frac 0.38 --suit_min 0.40 --rank_min 0.40 --debug

$ python read_indices_from_out_cards_v2.py --in_dir out_cards --roi_w 0.18 --roi_h 0.296 --split_y 0.598 --left_band 0.513 --max_rel_area 0.15 --min_area 18 --debug

python extract_crops_and_predict.py
