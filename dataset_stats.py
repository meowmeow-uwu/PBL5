import os
from config import DATASET_CACHUA_DIR, VAL_SIZE_FROM_TRAINVAL

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
    
    train_full_stats = count_samples(train_dir)
    test_stats = count_samples(test_dir)
    
    all_classes = sorted(list(set(list(train_full_stats.keys()) + list(test_stats.keys()))))
    
    print("\n" + "=" * 80)
    print("  THỐNG KÊ SỐ LƯỢNG MẪU TRONG TẬP DATASET CÀ CHUA")
    print("=" * 80)
    print(f"{'Tên lớp (Class)':<20} | {'Train':<10} | {'Val':<10} | {'Test':<10} | {'Tổng cộng':<10}")
    print("-" * 80)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for cls in all_classes:
        tr_full_count = train_full_stats.get(cls, 0)
        te_count = test_stats.get(cls, 0)
        
        # Calculate val based on config (approximate split)
        val_count = int(round(tr_full_count * VAL_SIZE_FROM_TRAINVAL))
        tr_count = tr_full_count - val_count
        
        total_train += tr_count
        total_val += val_count
        total_test += te_count
        
        total_cls = tr_count + val_count + te_count
        
        print(f"{cls:<20} | {tr_count:<10} | {val_count:<10} | {te_count:<10} | {total_cls:<10}")
        
    print("-" * 80)
    total_all = total_train + total_val + total_test
    print(f"{'TỔNG CỘNG':<20} | {total_train:<10} | {total_val:<10} | {total_test:<10} | {total_all:<10}")
    
    if total_all > 0:
        tr_pct = total_train / total_all * 100
        val_pct = total_val / total_all * 100
        te_pct = total_test / total_all * 100
        print(f"{'TỈ LỆ (%)':<20} | {tr_pct:<10.1f} | {val_pct:<10.1f} | {te_pct:<10.1f} | 100.0")
    print("=" * 80)

if __name__ == "__main__":
    main()
