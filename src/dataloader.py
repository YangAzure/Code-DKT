import torch
import torch.utils.data as Data
from readdata import data_reader


def get_data_loader(batch_size, num_of_questions, max_step, fold):
    handle = data_reader('../data/DKTFeatures/train_firstatt_'+str(fold)+'.csv',
                        '../data/DKTFeatures/val_firstatt_'+str(fold)+'.csv',
                         '../data/DKTFeatures/test_data.csv', max_step,
                        num_of_questions)


    dtrain = torch.tensor(handle.get_train_data().astype(float).tolist(),
                          dtype=torch.float32)
    dtest = torch.tensor(handle.get_test_data().astype(float).tolist(),
                         dtype=torch.float32)

    train_loader = Data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
