@echo off
call cd /D D:\wt\DATL
call conda activate pytorch
call python main_office.py --source amazon --target dslr
call python main_office.py --source amazon --target webcam
call python main_office.py --source dslr --target amazon
call python main_office.py --source dslr --target webcam
call python main_office.py --source webcam --target amazon
call python main_office.py --source webcam --target dslr
call python main_home.py --source Art --target Clipart
call python main_home.py --source Art --target Product
call python main_home.py --source Art --target Real_World
call python main_home.py --source Clipart --target Art
call python main_home.py --source Clipart --target Product
call python main_home.py --source Clipart --target Real_World
call python main_home.py --source Product --target Art
call python main_home.py --source Product --target Clipart
call python main_home.py --source Product --target Real_World
call python main_home.py --source Real_World --target Art
call python main_home.py --source Real_World --target Clipart
call python main_home.py --source Real_World --target Product
exit