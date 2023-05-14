import os
import sys
import argparse
from PIL import Image
from model import FaceClassifierResNet18
from torchvision import transforms
import torch

# 设置命令行参数
parser = argparse.ArgumentParser(description='Test the face classifier.')
parser.add_argument('--model_path', type=str, default=None, help='Path to the model file. If not provided, the latest model in the models directory will be used.')
parser.add_argument('--img_size', type=int, default=256, help="Image size (default: 256)")
parser.add_argument('--test_dir', type=str, default='./data/test', help="Data directory (default: './data/test')")
parser.add_argument('--face', type=str, default=1, help="The Images in test directory are faces or not.(1 or 0)")
args = parser.parse_args()
args.face = float(args.face)

# 如果未指定模型路径，则使用models文件夹中的最新模型
if args.model_path is None:
    model_dir = './models'
    model_files = os.listdir(model_dir)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    args.model_path = os.path.join(model_dir, model_files[0])

# 参数设置
test_data_dir = args.test_dir
img_size = args.img_size
threshold = 0.5

# 数据预处理
test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载测试数据
# print(os.listdir(test_data_dir))
test_image_paths = [os.path.join(test_data_dir, img) for img in os.listdir(test_data_dir)]
test_images = [Image.open(img_path).convert("RGB") for img_path in test_image_paths]
test_images = [test_transform(img) for img in test_images]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FaceClassifierResNet18()

model.load_state_dict(torch.load(args.model_path))

model.to(device)

model.eval()


# # 进行测试
# total = len(test_images)
# correct = 0
# with torch.no_grad():
#     for img in test_images:
#         img = img.unsqueeze(0).to(device)  # 添加批次维度并将图像转移到GPU
#         output = model(img)
#         prediction = (output > threshold).float().item()

#         if prediction == args.face:  # 如果预测是人脸
#             correct += 1

# accuracy = 100 * correct / total
# print(f"Test accuracy: {correct} / {total} = {accuracy:.2f}%")

# 计算真阳性率和假阳性率
def compute_tpr_fpr(preds, labels, threshold):
    TP, TN, FP, FN = 0, 0, 0, 0

    for pred, label in zip(preds, labels):
        prediction = (pred > threshold).float().item()

        if prediction == 1.0 and label == 1.0:
            TP += 1
        elif prediction == 1.0 and label == 0.0:
            FP += 1
        elif prediction == 0.0 and label == 1.0:
            FN += 1
        elif prediction == 0.0 and label == 0.0:
            TN += 1

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    return TPR, FPR

# 计算模型在测试数据上的预测结果
with torch.no_grad():
    preds = [model(img.unsqueeze(0).to(device)) for img in test_images]

# 计算ROC曲线的数据
thresholds = np.linspace(0, 1, 100)
tpr_list, fpr_list = [], []

for threshold in thresholds:
    tpr, fpr = compute_tpr_fpr(preds, test_labels, threshold)
tpr_list.append(tpr)
fpr_list.append(fpr)


# 计算AUC（Area Under Curve）
roc_auc = auc(fpr_list, tpr_list)
#绘制ROC曲线
plt.figure()
plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()