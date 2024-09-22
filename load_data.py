import random


def read_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        lines = file.readlines()
    texts = []
    image = []
    images = []
    for line in lines:
        line = line.strip()
        if line.startswith('text: '):
            text = line.split('text: ')[1]
            texts.append(text)
        if line.endswith(".jpg") or line.endswith(".png"):
            image.append(line)
        if line.startswith("id: ") and len(image) != 0:
            images.append(image)
            image = []
    images.append(image)
    return texts, images


def process_image_path(text_list, image_list):
    positive_path = []
    negative_path = []
    texts = []
    for i in range(len(image_list)):
        image_path = image_list[i]
        idx = 0
        for j in range(len(image_path)):
            if image_path[j] == "inf.png":
                idx = j
                break
        if idx > 0:
            random_number = random.randint(0, idx - 1)
            positive_path.append(image_path[random_number])
            negative_path.append(image_path[idx])
            texts.append(text_list[i])
        if idx < len(image_path) - 1:
            random_number = random.randint(idx + 1, len(image_path) - 1)
            positive_path.append(image_path[idx])
            negative_path.append(image_path[random_number])
            texts.append(text_list[i])
    positive_paths = []
    negative_paths = []
    for e in positive_path:
        if e.startswith("twitter"):
            result = "".join(["./dataset/twitter2017_images/", e.split('-')[1]])
        else:
            result = "".join(["./dataset/new_images/", e])
        positive_paths.append(result)
    for e in negative_path:
        if e.startswith("twitter"):
            result = "".join(["./dataset/twitter2017_images/", e.split('-')[1]])
        else:
            result = "".join(["./dataset/new_images/", e])
        negative_paths.append(result)
    return positive_paths, negative_paths, texts


def process_image_path_eval(text_list, image_list):
    positive_path = []
    negative_path = []
    texts = []
    for i in range(len(image_list)):
        images = image_list[i]
        for j in range(len(images)):
            positive_path.append(images[j])
            negative_path.append("inf.png")
            texts.append(text_list[i])
    positive_paths = []
    negative_paths = []
    for e in positive_path:
        if e.startswith("twitter"):
            result = "".join(["./dataset/twitter2017_images/", e.split('-')[1]])
        else:
            result = "".join(["./dataset/new_images/", e])
        positive_paths.append(result)
    for e in negative_path:
        if e.startswith("twitter"):
            result = "".join(["./dataset/twitter2017_images/", e.split('-')[1]])
        else:
            result = "".join(["./dataset/new_images/", e])
        negative_paths.append(result)
    return positive_paths, negative_paths, texts


def process_image_path_all(text_list, image_list):
    positive_path = []
    negative_path = []
    texts = []
    for i in range(len(image_list)):
        images = image_list[i]
        for j in range(len(images) - 1):
            positive_path.append(images[j])
            negative_path.append(images[j + 1])
            texts.append(text_list[i])
    positive_paths = []
    negative_paths = []
    for e in positive_path:
        if e.startswith("twitter"):
            result = "".join(["./dataset/twitter2017_images/", e.split('-')[1]])
        else:
            result = "".join(["./dataset/new_images/", e])
        positive_paths.append(result)
    for e in negative_path:
        if e.startswith("twitter"):
            result = "".join(["./dataset/twitter2017_images/", e.split('-')[1]])
        else:
            result = "".join(["./dataset/new_images/", e])
        negative_paths.append(result)
    return positive_paths, negative_paths, texts


def write_reward(file_path, reward):
    with open(file_path, 'r', encoding="utf-8") as file:
        lines = file.readlines()
    with open("./dataset/text/reward_data.txt", "w", encoding="utf-8") as output_file:
        idx = 0
        cnt = 0
        for line in lines:
            line = line.strip()
            if line.endswith(".jpg") or line.endswith(".png"):
                cnt += 1
            if line.startswith("id: ") and cnt != 0:
                for i in range(cnt):
                    output_file.write(str(reward[idx]))
                    output_file.write('\n')
                    idx += 1
                cnt = 0
            output_file.write(line)
            output_file.write('\n')
        for i in range(cnt):
            output_file.write(str(reward[idx]))
            output_file.write('\n')
            idx += 1
