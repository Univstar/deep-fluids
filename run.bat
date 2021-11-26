REM ----- smoke_pos_size, 2D
..\manta\build\Release\manta.exe .\scene\smoke_pos_size.py
python main.py
python main.py --is_train=False --load_path=MODEL_DIR

REM ----- liquid_pos_size, 2D
..\manta\build\Release\manta.exe .\scene\liquid_pos_size.py
python main.py --use_curl=False --dataset=liquid_pos10_size4_f200 --res_x=128 --res_y=64
python main.py --is_train=False --load_path=MODEL_DIR --dataset=liquid_pos10_size4_f200 --res_x=128 --res_y=64
