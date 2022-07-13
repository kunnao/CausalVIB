CALL D:\Anaconda\Scripts\activate.bat D:\Anaconda\envs\tensorflow

cd D:\study_research\Causal Effect Inference\causal effect tool kit

python train_main.py --network_type causalvib --dataset twin --targeted_regularization 0 --batch_size 256 --replication 20 --auto 0
python train_main.py --network_type causalvib --dataset ihdp --targeted_regularization 0 --batch_size 32 --replication 20
python train_main.py --network_type causalvib --dataset acic --targeted_regularization 0 --batch_size 256 --replication 5


