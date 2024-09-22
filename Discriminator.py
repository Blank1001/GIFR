import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from clip import clip
from torch.utils.data import DataLoader

from load_data import read_file, process_image_path_eval, write_reward, process_image_path_all

file_path = "./dataset/text/preference2.txt"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)

model.requires_grad_(True)


class RewardModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

def init_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.kaiming_normal_(param.data)
        elif "bias" in name:
            nn.init.constant_(param.data, 0.0)


def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        image = preprocess(image).unsqueeze(0).to(device)
    except OSError as e:
        image = Image.open("./dataset/new_images/inf.png")
        image = preprocess(image).unsqueeze(0).to(device)
    return image


def preprocess_text(text):
    text = clip.tokenize(text).to(device)
    return text


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, positive_list, negative_list, texts):
        self.positive_list = positive_list
        self.negative_list = negative_list
        self.texts = texts

    def __getitem__(self, index):
        positive_image_path = self.positive_list[index]
        negative_image_path = self.negative_list[index]
        text = self.texts[index]

        positive_image = preprocess_image(positive_image_path)
        negative_image = preprocess_image(negative_image_path)
        text = preprocess_text(text)

        return positive_image, negative_image, text

    def __len__(self):
        return len(self.positive_list)


def dev_eval(eval_dataloader):

    mlp_model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for positive_images, negative_images, texts in eval_dataloader:

            positive_images = positive_images.to(device)
            negative_images = negative_images.to(device)
            texts = texts.to(device)

            positive_images = torch.squeeze(positive_images, dim=1)
            negative_images = torch.squeeze(negative_images, dim=1)
            texts = torch.squeeze(texts, dim=1)

            positive_image_features = model.encode_image(positive_images)
            negative_image_features = model.encode_image(negative_images)
            text_features = model.encode_text(texts)

            positive_input_features = torch.cat((positive_image_features, text_features), dim=1).float()

            negative_input_features = torch.cat((negative_image_features, text_features), dim=1).float()


            positive_rewards = mlp_model(positive_input_features)
            negative_rewards = mlp_model(negative_input_features)

            loss = -torch.mean(torch.log(torch.sigmoid(positive_rewards - negative_rewards)))

            eval_loss += loss.item()


    eval_loss /= len(eval_dataloader)

    mlp_model.train()


    print(f"Evaluation Loss: {eval_loss:.4f}")
    return eval_loss


def test_eval():
    reward = []
    eval_text, eval_image = read_file("./dataset/text/data_UNI.txt")
    positive_paths_eval, negative_paths_eval, texts_eval = process_image_path_eval(eval_text, eval_image)
    test_dataset = CustomDataset(positive_paths_eval, negative_paths_eval, texts_eval)


    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for positive_images, negative_images, texts in test_dataloader:

            positive_images = positive_images.to(device)
            negative_images = negative_images.to(device)
            texts = texts.to(device)

            positive_images = torch.squeeze(positive_images, dim=1)
            negative_images = torch.squeeze(negative_images, dim=1)
            texts = torch.squeeze(texts, dim=1)

            positive_image_features = model.encode_image(positive_images)
            negative_image_features = model.encode_image(negative_images)
            text_features = model.encode_text(texts)

            positive_input_features = torch.cat((positive_image_features, text_features), dim=1).float()
            negative_input_features = torch.cat((negative_image_features, text_features), dim=1).float()


            positive_rewards = mlp_model(positive_input_features)
            negative_rewards = mlp_model(negative_input_features)


            results = torch.sigmoid(positive_rewards - negative_rewards)
            for result in results:
                reward.append(result.item())

    write_reward("./dataset/text/data_UNI.txt", reward)

text_list, image_list = read_file(file_path)

positive_image_paths, negative_image_paths, texts = process_image_path_all(text_list, image_list)
eval_positive_image_paths = positive_image_paths[-100:]
eval_negative_image_paths = negative_image_paths[-100:]
eval_texts = texts[-100:]
positive_image_paths = positive_image_paths[:-100]
negative_image_paths = negative_image_paths[:-100]
texts = texts[:-100]

dataset = CustomDataset(positive_image_paths, negative_image_paths, texts)

batch_size = 8
num_epochs = 10

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

input_size = 1024
hidden_size = 256
mlp_model = RewardModel(input_size, hidden_size).to(device)

optimizer = optim.Adam(mlp_model.parameters(), lr=0.0001, weight_decay=0.001)

eval_dataset = CustomDataset(eval_positive_image_paths, eval_negative_image_paths, eval_texts)

eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

init_weights(mlp_model)
min_loss = 1000
for epoch in range(num_epochs):
    running_loss = 0.0
    for positive_images, negative_images, texts in dataloader:
        # [8, 1, 3, 224, 224]
        positive_images = positive_images.to(device)
        negative_images = negative_images.to(device)
        texts = texts.to(device)
        # [8, 3, 224, 224]
        positive_images = torch.squeeze(positive_images, dim=1)
        negative_images = torch.squeeze(negative_images, dim=1)
        texts = torch.squeeze(texts, dim=1)
        positive_image_features = model.encode_image(positive_images)
        negative_image_features = model.encode_image(negative_images)
        text_features = model.encode_text(texts)
        # [8, 1024]
        positive_input_features = torch.cat((positive_image_features, text_features), dim=1).float()
        negative_input_features = torch.cat((negative_image_features, text_features), dim=1).float()

        positive_rewards = mlp_model(positive_input_features)
        negative_rewards = mlp_model(negative_input_features)

        loss = -torch.mean(torch.log(torch.sigmoid(positive_rewards - negative_rewards)))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    eval_loss = dev_eval(eval_dataloader)
    if eval_loss < min_loss and epoch > 5:
        test_eval()
    min_loss = min(min_loss, eval_loss)
