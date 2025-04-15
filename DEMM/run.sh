python main.py  --L 5 --alpha 4 --dataset acm-3025 --gamma 0. --dim 128  --seed 6  --beta 2.5 --method demm+ --m 10 14  --gpu 0
python main.py  --L 6 --alpha 28 --dataset dblp --gamma 0. --dim 64 --seed 42 --max_h 2 --beta 25 --method demm+ --m 10 8 10 --gpu 0
python main.py  --L 3 --alpha 4 --dataset acm-4019 --gamma 0. --dim 512  --seed 6  --beta 4.2 --method demm+ --m 10 10  --gpu 0
python main.py  --L 16 --alpha 32 --dataset yelp --gamma 0. --dim 32 --seed 6 --beta 3 --method demm+ --m 14 12 16 --gpu 0
python main.py  --L 13 --alpha 7 --dataset imdb --gamma 0. --dim 1024  --seed 6 --beta 6 --method demm+ --m 16 16  --gpu 0
python large.py --L 14 --alpha 50 --dataset mag --gamma 0. --dim 32 --seed 42 --gpu 0 --beta 20 --m 12 12  --method demm+
python large.py --L 16 --alpha 120  --dataset oag-eng --gamma 0. --dim 128 --seed 42 --gpu 0 --method demm+ --m 40 40 40 --beta 90
python large.py --L 12 --alpha 110  --dataset oag-cs --gamma 0. --dim 128 --seed 6 --gpu 1 --method demm+ --m 36 36 36 --beta 70

python demm-main.py  --alpha 2 --dataset acm-3025 --gamma 0. --seed 6 --beta 2 --gpu 0
python demm-main.py  --alpha 1900  --dataset dblp --gamma 0. --seed 42 --beta 4200 --gpu 0
python demm-main.py  --alpha 1.5 --dataset acm-4019 --gamma 0. --seed 6 --beta 2 --gpu 0
python demm-main.py  --alpha 26 --dataset yelp --gamma 0. --seed 6 --beta 50 --gpu 0
python demm-main.py  --alpha 6 --dataset imdb --gamma 0.  --seed 6 --beta 8 --gpu 0
python demm-large.py  --alpha 50 --dataset mag --gamma 0.  --seed 42 --beta 6 --gpu 0

python main.py   --dataset acm-3025 --gamma 0. --dim 6  --seed 6  --beta 2 --method demmal --m 10 10 --gpu 0
python main.py   --dataset dblp --gamma 0. --dim 4  --seed 6  --beta 25 --method demmal --m 10 10 10 --gpu 0
python main.py   --dataset acm-4019 --gamma 0. --dim 4  --seed 6  --beta 2 --method demmal --m 10 10 --gpu 0
python main.py   --dataset yelp --gamma 0. --dim 3  --seed 6  --beta 24 --method demmal --m 10 10 10 --gpu 0
python main.py   --dataset imdb --gamma 0. --dim 80  --seed 6  --beta 10 --method demmal --m 16 16 --gpu 0
python large.py   --dataset oag-cs --gamma 0. --dim 68  --seed 6  --beta 280 --method demmal --m 36 36 36 --gpu 0
python large.py   --dataset oag-eng --gamma 0. --dim 62  --seed 6  --beta 340 --method demmal --m 36 36 36 --gpu 0