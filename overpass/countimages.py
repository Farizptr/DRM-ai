import os
from pathlib import Path

def count_images_in_split(split_path: Path) -> int:
    """
    Count number of images in a specific split directory
    
    Args:
        split_path (Path): Path to split directory
        
    Returns:
        int: Number of images found
    """
    if not split_path.exists():
        print(f"{split_path.name} folder not found.")
        return 0
        
    image_extensions = ('.png', '.jpg', '.jpeg')
    return len([f for f in split_path.iterdir() if f.suffix.lower() in image_extensions])

def count_dataset_images(dataset_dir: str = "kumpulandatasets") -> dict:  # Changed default directory name
    """
    Count images in all splits of the dataset
    
    Args:
        dataset_dir (str): Path to dataset directory
        
    Returns:
        dict: Dictionary containing counts for each split and total
    """
    base_path = Path(dataset_dir) / "images"
    splits = ["train", "val", "test"]
    counts = {}
    
    for split in splits:
        split_path = base_path / split
        count = count_images_in_split(split_path)
        counts[split] = count
        print(f"{split.capitalize()} images: {count}")
    
    total = sum(counts.values())
    counts["total"] = total
    print(f"Total images: {total}")
    
    return counts

if __name__ == "__main__":
    count_dataset_images()