for i in 1 2 3 4 5
        do python3.5 NN_CSS_cuda1.py --epochs 100 --seed $i --dropout 0.5
done

for i in 1 2 3 4 5
        do python3.5 NN_CSS_cuda1.py --epochs 100 --seed $i --dropout 0.0
done

for i in 1 2 3 4 5
        do python3.5 NN_CSS_cuda1.py --epochs 200 --seed $i --dropout 0.5
done

for i in 1 2 3 4 5
        do python3.5 NN_CSS_cuda1.py --epochs 200 --seed $i --dropout 0.0
done


for i in 1 2 3 4 5
        do python3.5 NN_CSS_cuda1.py --epochs 50 --seed $i --dropout 0.5
done

for i in 1 2 3 4 5
        do python3.5 NN_CSS_cuda1.py --epochs 50 --seed $i --dropout 0.0
done


