import os
from config import DATASET_CACHUA_DIR

def count_samples(directory):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return {}
    
    stats = {}
    for class_name in sorted(os.listdir(directory)):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            num_samples = len([
                f for f in os.listdir(class_dir) 
                if os.path.isfile(os.path.join(class_dir, f)) and 
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
            ])
            stats[class_name] = num_samples
    return stats

def main():
    train_dir = os.path.join(DATASET_CACHUA_DIR, "train")
    test_dir = os.path.join(DATASET_CACHUA_DIR, "test")
    
    print(f"Analyzing dataset at: {DATASET_CACHUA_DIR}")
    
    train_stats = count_samples(train_dir)
    test_stats = count_samples(test_dir)
    
    all_classes = sorted(list(set(list(train_stats.keys()) + list(test_stats.keys()))))
    
    print("\n" + "=" * 60)
    print("  THỐNG KÊ SỐ LƯỢNG MẪU TRONG TẬP DATASET CÀ CHUA")
    print("=" * 60)
    print(f"{'Tên lớp (Class)':<25} | {'Train':<10} | {'Test':<10} | {'Tổng cộng':<10}")
    print("-" * 60)
    
    total_train = 0
    total_test = 0
    
    for cls in all_classes:
        tr_count = train_stats.get(cls, 0)
        te_count = test_stats.get(cls, 0)
        total_train += tr_count
        total_test += te_count
        
        print(f"{cls:<25} | {tr_count:<10} | {te_count:<10} | {tr_count + te_count:<10}")
        
    print("-" * 60)
    print(f"{'TỔNG CỘNG':<25} | {total_train:<10} | {total_test:<10} | {total_train + total_test:<10}")
    print("=" * 60)

if __name__ == "__main__":
    main()
