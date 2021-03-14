import os
main_path = "/content/drive/My Drive/DIV25Dataset/Dataset"

hr_train_path = os.path.join(main_path, "DIV2K_train_HR")
hr_valid_path = os.path.join(main_path, "DIV2K_valid_HR")

lr_train_path = os.path.join(main_path, "DIV2K_train_LR_bicubic/X2")
lr_valid_path = os.path.join(main_path, "DIV2K_valid_LR_bicubic/X2")

best_model_path = '/content/drive/MyDrive/DIV25Dataset/MAANet_Models/'
best_model_name = 'best_model.pt'