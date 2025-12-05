from scipy.io import loadmat

data = loadmat('code/school_work/ECE_569/HW_3/train_overlap.mat')
print('train_overlap:', data.keys())
print('train_overlap A shape:', data['A'].shape)
print('train_overlap B shape:', data['B'].shape)

data = loadmat('code/school_work/ECE_569/HW_3/test_overlap.mat')
print('test_overlap', data.keys())
print('test_overlap X_test shape:', data['X_test'].shape)

data = loadmat('code/school_work/ECE_569/HW_3/train_separable.mat')
print('train_separable', data.keys())
print('train_separable A shape:', data['A'].shape)
print('train_separable B shape:', data['B'].shape)

data = loadmat('code/school_work/ECE_569/HW_3/test_separable.mat')
print('test_separable', data.keys())
print('test_separable X_test shape:', data['X_test'].shape)