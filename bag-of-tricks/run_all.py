import os

epoch = 200
trials = 3
C10 = "cifar10"
C100 = "cifar100"
AS = "asymmetric"
SY = "symmetric"
IN = "instance"
AGG = "aggre"
WST = "worst"
N100 = "noisy100"


mixup = f"python main.py --dataset {C10} --epoch {epoch} --gpu 1 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {SY} --noise_rate 0.2 & \
          python main.py --dataset {C10} --epoch {epoch} --gpu 2 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {SY} --noise_rate 0.4 & \
          python main.py --dataset {C10} --epoch {epoch} --gpu 3 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {SY} --noise_rate 0.6 & \
          python main.py --dataset {C10} --epoch {epoch} --gpu 0 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {AS} --noise_rate 0.4 & \
          python main.py --dataset {C10} --epoch {epoch} --gpu 1 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {IN} --noise_rate 0.2 & \
          python main.py --dataset {C10} --epoch {epoch} --gpu 2 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {IN} --noise_rate 0.4 & \
          python main.py --dataset {C10} --epoch {epoch} --gpu 3 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {AGG} & \
          python main.py --dataset {C10} --epoch {epoch} --gpu 0 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {WST} & \
          python main.py --dataset {C100} --epoch {epoch} --gpu 1 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {SY} --noise_rate 0.2 & \
          python main.py --dataset {C100} --epoch {epoch} --gpu 2 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {SY} --noise_rate 0.4 & \
          python main.py --dataset {C100} --epoch {epoch} --gpu 3 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {SY} --noise_rate 0.6 & \
          python main.py --dataset {C100} --epoch {epoch} --gpu 0 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {AS} --noise_rate 0.4 & \
          python main.py --dataset {C100} --epoch {epoch} --gpu 1 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {IN} --noise_rate 0.2 & \
          python main.py --dataset {C100} --epoch {epoch} --gpu 2 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {IN} --noise_rate 0.4 & \
          python main.py --dataset {C100} --epoch {epoch} --gpu 3 --trials {trials} --alpha 1\
             --lr 0.02 --noise_type {N100}"


baseline = f"python main.py --dataset {C10} --epoch {epoch} --gpu 0 --trials {trials} \
              --lr 0.02 --noise_type {AS} --noise_rate 0.4 & \
            python main.py --dataset {C100} --epoch {epoch} --gpu 1 --trials {trials} \
              --lr 0.02 --noise_type {AS} --noise_rate 0.4 & \
            python main.py --dataset {C10} --epoch {epoch} --gpu 2 --trials {trials} \
              --lr 0.02 --noise_type {SY} --noise_rate 0.6" 


label_smoothing = f"python main.py --dataset {C10} --epoch {epoch} --gpu 3 --trials {trials} --label_smoothing 0.1\
              --lr 0.02 --noise_type {AS} --noise_rate 0.4 & \
            python main.py --dataset {C100} --epoch {epoch} --gpu 0 --trials {trials} --label_smoothing 0.1\
              --lr 0.02 --noise_type {AS} --noise_rate 0.4 & \
            python main.py --dataset {C10} --epoch {epoch} --gpu 1 --trials {trials} --label_smoothing 0.1\
              --lr 0.02 --noise_type {SY} --noise_rate 0.6"

autoaug = f"python main.py --dataset {C10} --epoch {epoch} --gpu 2 --trials {trials}  --aug_type 'autoaug' --consistency 1\
              --lr 0.02 --noise_type {AS} --noise_rate 0.4 & \
            python main.py --dataset {C100} --epoch {epoch} --gpu 3 --trials {trials}  --aug_type 'autoaug' --consistency 1\
              --lr 0.02 --noise_type {AS} --noise_rate 0.4 & \
            python main.py --dataset {C10} --epoch {epoch} --gpu 0 --trials {trials}  --aug_type 'autoaug' --consistency 1\
              --lr 0.02 --noise_type {SY} --noise_rate 0.6" 

randaug = f"python main.py --dataset {C10} --epoch {epoch} --gpu 1 --trials {trials}  --aug_type 'randaug' --consistency 1\
              --lr 0.02 --noise_type {AS} --noise_rate 0.4 & \
            python main.py --dataset {C100} --epoch {epoch} --gpu 2 --trials {trials}  --aug_type 'randaug' --consistency 1\
              --lr 0.02 --noise_type {AS} --noise_rate 0.4 & \
            python main.py --dataset {C10} --epoch {epoch} --gpu 3 --trials {trials}  --aug_type 'randaug' --consistency 1\
              --lr 0.02 --noise_type {SY} --noise_rate 0.6" 

class_balance = f"python main.py --dataset {C10} --epoch {epoch} --gpu 0 --trials {trials}  --class_balance 1\
              --lr 0.02 --noise_type {AS} --noise_rate 0.4 & \
              python main.py --dataset {C100} --epoch {epoch} --gpu 1 --trials {trials}  --class_balance 1\
              --lr 0.02 --noise_type {AS} --noise_rate 0.4 & \
            python main.py --dataset {C10} --epoch {epoch} --gpu 2 --trials {trials}  --class_balance 1\
              --lr 0.02 --noise_type {SY} --noise_rate 0.6" 


os.system(mixup)
os.system(f"{baseline} & {label_smoothing} & {autoaug} & {randaug} & {class_balance}")
