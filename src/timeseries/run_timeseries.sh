hidden_ls=(8 16)
lr_ls=(1e-7 1e-8)
for i in ${hidden_ls[*]}; do
    for j in ${lr_ls[*]}; do
        python train.py --hidden_dim $i --lr $j
    done
done