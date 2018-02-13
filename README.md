# MetaNN
MetaNN consists of two directory: DataSet and Evaluation
1. DataSet directory:
	- Consists of 8 real datasets that are used to evaluate MetaNN
	- For each dataset, it consists of two subdirectory: DataSet/Code and DataSet/Data
		- Code/classifier: implementation of ML classifiers and NN classifiers
		- Code/scripts: scripts for data augmentation and dataset splits
		- Data/Augmentation: augmented training dataset
		- Data/Results_CNN: testing result for CNN
		- Data/Results_MLP: testing result for MLP
		- Data/Results: evaluation results of all the classifiers including NN classifiers
		- Data/Splits: training-testing splits
2. Evaluation directory:
	- F1 scores and ROC curves
3. Usage: IBD dataset as an example
	- Data augmentation and splits: IBD_Aug.R and IBD_Splits.py
	- Run NN classifier: `python3 NN_IBD_cuda.py`
	- Run ML classifier: `python classifier_IBD_all.py --dataset IBD --nc 2`
	- Run evaluation: `python print_stats_IBD.py --dataset IBD --nc 2`
