import numpy as np
from sklearn.utils import resample


def extract_samples(X_train, y_train, X_test, y_test, new_app, k, j):
    """
    从 y_test 中提取类别为 k 的 k 个样本，从 X_train 中提取除类别 k 外的 j 个样本。

    参数:
        X_train (ndarray): 训练特征数据，形状为 (n_train_samples, n_features)
        y_train (ndarray): 训练标签数据，形状为 (n_train_samples,)
        X_test (ndarray): 测试特征数据，形状为 (n_test_samples, n_features)
        y_test (ndarray): 测试标签数据，形状为 (n_test_samples,)
        k (int): 从 y_test 中类别为 k 的样本数量
        j (int): 从 X_train 中非类别 k 的样本数量

    返回:
        X_tune (ndarray): 提取的特征数据，形状为 (k + j, n_features)
        y_tune (ndarray): 提取的标签数据，形状为 (k + j,)
    """
    # 从 y_test 中找到类别为 k 的样本索引
    test_indices_k = np.where(y_test == new_app)[0]

    # 检查是否有足够的类别 k 的样本，随机选择 k 个样本
    if len(test_indices_k) >= k:
        chosen_test_indices = np.random.choice(test_indices_k, k, replace=False)
    else:
        raise ValueError(f"y_test 中类别 {new_app} 的样本不足 {k} 个。")

    # 获取所有类别
    unique_classes = np.unique(y_train)
    # 排除类别 k
    classes_except_k = [cls for cls in unique_classes if cls != new_app]

    # 存储选取的样本
    selected_X = []
    selected_y = []

    for cls in classes_except_k:
        # 找到当前类别的所有样本的索引
        indices_cls = np.where(y_train == cls)[0]

        # 检查当前类别是否有足够的样本
        if len(indices_cls) >= j:
            # 随机选择 j 个样本
            chosen_indices = np.random.choice(indices_cls, j, replace=False)
            selected_X.extend(X_train[chosen_indices])
            selected_y.extend(y_train[chosen_indices])
        else:
            raise ValueError(f"y_train 中类别 {cls} 的样本不足 {j} 个。")

    # 转换为 ndarray 格式
    selected_X = np.array(selected_X)
    selected_y = np.array(selected_y)


    # 从 X_test 和 y_test 中提取类别为 k 的样本
    X_tune_k = X_test[chosen_test_indices]
    y_tune_k = y_test[chosen_test_indices]

    # 合并结果
    X_tune = np.concatenate((X_tune_k, selected_X), axis=0)
    y_tune = np.concatenate((y_tune_k, selected_y), axis=0)

    return X_tune, y_tune


def extract_target_samples(X_test, y_test, new_app, k):

    # 从 y_test 中找到类别为 k 的样本索引
    test_indices_k = np.where(y_test == new_app)[0]

    # 检查是否有足够的类别 k 的样本，随机选择 k 个样本
    if len(test_indices_k) >= k:
        chosen_test_indices = np.random.choice(test_indices_k, k, replace=False)
    else:
        raise ValueError(f"y_test 中类别 {new_app} 的样本不足 {k} 个。")

    # 从 X_test 和 y_test 中提取类别为 k 的样本
    X_tune_k = X_test[chosen_test_indices]
    y_tune_k = y_test[chosen_test_indices]

    return X_tune_k, y_tune_k


def plaid_data_augmentation(X, y, min_samples_per_class=20, random_state=42):
    """
    扩充少数类别的样本数量，使得每个类别的样本数量至少达到指定的数量。

    参数:
    - X: ndarray, 样本数据，形状为 (n_samples, n_features)
    - y: ndarray, 样本对应的标签，形状为 (n_samples,)
    - min_samples_per_class: int, 每个类别的最小样本数
    - random_state: int, 随机种子，确保结果可复现

    返回:
    - X_augmented: ndarray, 扩充后的样本数据
    - y_augmented: ndarray, 扩充后的样本标签
    """
    # 获取每个类别的标签
    unique_labels = np.unique(y)
    X_augmented, y_augmented = [], []

    # 遍历每个类别
    for label in unique_labels:
        # 提取该类别的样本
        X_class = X[y == label]
        y_class = y[y == label]

        # 如果该类别的样本数量不足 min_samples_per_class，则进行扩充
        if len(X_class) < min_samples_per_class:
            X_class_resampled, y_class_resampled = resample(
                X_class, y_class,
                replace=True,  # 允许重复抽样
                n_samples=min_samples_per_class,  # 扩充到 min_samples_per_class 个样本
                random_state=random_state  # 固定随机种子
            )
        else:
            X_class_resampled, y_class_resampled = X_class, y_class

        # 将扩充后的样本添加到最终数据集中
        X_augmented.append(X_class_resampled)
        y_augmented.append(y_class_resampled)

    # 合并所有类别的数据
    X_augmented = np.vstack(X_augmented)
    y_augmented = np.concatenate(y_augmented)

    return X_augmented, y_augmented


def construct_house_data(X_input, y_label, house_label, test_size=0.5, random_seed=None):
    """
    构建两个不重叠的抽象 house 数据集，确保每个标签类别的数据只在一个 house 中存在。

    参数：
    - X_input: ndarray, 输入特征数据
    - y_label: ndarray, 数据标签
    - house_label: ndarray, 每条数据的房屋标签
    - test_size: float, house2_data 中数据的比例，默认为 0.5
    - random_seed: int, 随机种子，用于控制分割的随机性

    返回：
    - house1_data: dict, 包含 house1 的数据 {'X': ..., 'y': ..., 'house': ...}
    - house2_data: dict, 包含 house2 的数据 {'X': ..., 'y': ..., 'house': ...}
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    unique_labels = np.unique(y_label)  # 获取所有标签类别
    house1_indices = []
    house2_indices = []
    used_labels = set()  # 记录已经分配的标签

    # 遍历每个标签类别
    for label in unique_labels:
        # 获取当前标签的数据索引
        label_indices = np.where(y_label == label)[0]

        # 选择当前标签数据对应的房屋标签
        label_house_labels = house_label[label_indices]

        # 获取当前标签的所有房屋
        unique_houses_for_label = np.unique(label_house_labels)

        # 随机选择一个房屋分配给 house1，剩下的分配给 house2
        selected_houses = np.random.choice(unique_houses_for_label, 2, replace=False)
        selected_house1 = selected_houses[0]
        selected_house2 = selected_houses[1]

        # 获取分配给 house1 的数据索引
        house1_label_indices = label_indices[house_label[label_indices] == selected_house1]
        house2_label_indices = label_indices[house_label[label_indices] == selected_house2]

        # 添加到 house1 和 house2 的索引集合中
        house1_indices.extend(house1_label_indices)
        house2_indices.extend(house2_label_indices)

    # 构建 house1_data 和 house2_data
    house1_data = {
        'X': X_input[house1_indices],
        'y': y_label[house1_indices],
        'house': house_label[house1_indices]
    }

    house2_data = {
        'X': X_input[house2_indices],
        'y': y_label[house2_indices],
        'house': house_label[house2_indices]
    }

    return house1_data, house2_data


def data_augmentation(X, y, N_f, lambda_coeff=3.0):
    """
    Perform data augmentation for each class in the dataset.

    Parameters:
    - X: numpy array of shape (num_samples, num_features), the original feature data
    - y: numpy array of shape (num_samples,), the labels
    - N_f: int, number of augmented samples per class
    - lambda_coeff: float, augmentation coefficient to scale standard deviation

    Returns:
    - X_augmented: numpy array of shape (num_samples + num_augmented, num_features),
      the original and augmented data combined
    - y_augmented: numpy array of shape (num_samples + num_augmented,), labels including
      original and augmented data
    """
    unique_classes = np.unique(y)
    X_augmented = []
    y_augmented = []

    for cls in unique_classes:
        # Filter data for the current class
        X_class = X[y == cls]

        # Calculate mean and std for each feature within this class
        means = np.mean(X_class, axis=0)
        std_devs = np.std(X_class, axis=0)

        # Generate augmented data for this class
        for _ in range(N_f):
            noisy_sample = np.random.normal(means, lambda_coeff * std_devs)
            X_augmented.append(noisy_sample)
            y_augmented.append(cls)  # Assign the class label to the augmented sample

    # Convert lists to numpy arrays
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)

    # Combine original data with augmented data
    X_combined = np.vstack((X, X_augmented))
    y_combined = np.concatenate((y, y_augmented))

    return X_combined, y_combined

