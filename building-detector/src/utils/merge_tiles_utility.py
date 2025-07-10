#!/usr/bin/env python3
"""
Utility script untuk menggabungkan semua file JSON tiles menjadi satu file JSON
dengan format sederhana: id (increment dari 1), longitude, latitude
"""

import sys
import os
from ..core.polygon_detection import merge_all_tiles_to_simple_json

def _print_header():
    """Print utility header information"""
    print("="*60)
    print("TILES MERGER UTILITY")
    print("="*60)
    print("Menggabungkan semua file JSON tiles menjadi satu file")
    print("Format output: [{\"id\": 1, \"longitude\": 106.123, \"latitude\": -6.456}, ...]")
    print("="*60)

def _print_usage():
    """Print usage instructions"""
    print("Usage:")
    print("  python merge_tiles_utility.py <tiles_directory> [output_file]")
    print("  python merge_tiles_utility.py @tiles [output_file]")
    print("")
    print("Examples:")
    print("  python merge_tiles_utility.py polygon_detection_results/tiles")
    print("  python merge_tiles_utility.py polygon_detection_results merged_buildings.json")
    print("  python merge_tiles_utility.py @tiles buildings_simple.json")
    print("")
    print("Jika @tiles digunakan, akan mencari folder 'tiles' di direktori saat ini")

def _parse_arguments():
    """Parse command line arguments and return tiles_input and output_file"""
    if len(sys.argv) < 2:
        _print_usage()
        return None, None
    
    tiles_input = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else "merged_buildings_simple.json"
    
    return tiles_input, output_file

def _resolve_tiles_directory(tiles_input):
    """Resolve tiles directory from input, handling @tiles shortcut"""
    if tiles_input == "@tiles":
        tiles_dir = "./tiles"
        if not os.path.exists(tiles_dir):
            # Try looking in common output directories
            possible_dirs = [
                "./polygon_detection_results/tiles",
                "../polygon_detection_results/tiles",
                "./results/tiles"
            ]
            for possible_dir in possible_dirs:
                if os.path.exists(possible_dir):
                    tiles_dir = possible_dir
                    break
        
        print(f"Using @tiles shortcut, searching in: {tiles_dir}")
        return tiles_dir
    else:
        return tiles_input

def _validate_tiles_directory(tiles_dir):
    """Validate that tiles directory exists"""
    if not os.path.exists(tiles_dir):
        print(f"❌ Error: Tiles directory not found: {tiles_dir}")
        print("")
        print("Tips:")
        print("- Pastikan path tiles directory benar")
        print("- Gunakan @tiles untuk auto-detect dari direktori umum")
        print("- Jalankan detection script terlebih dahulu untuk menghasilkan tiles")
        return False
    return True

def _perform_merge_operation(tiles_dir, output_file):
    """Perform the merge operation and display results"""
    try:
        result_count = merge_all_tiles_to_simple_json(tiles_dir, output_file)
        
        if result_count > 0:
            _display_success_results(result_count, output_file)
        else:
            _display_failure_results()
            
    except Exception as e:
        print(f"\n❌ ERROR during merge: {e}")

def _display_success_results(result_count, output_file):
    """Display successful merge results"""
    print("\n🎉 MERGE COMPLETED SUCCESSFULLY!")
    print(f"📊 Total buildings merged: {result_count}")
    print(f"📁 Output file: {output_file}")
    
    # Show file size
    try:
        file_size = os.path.getsize(output_file)
        print(f"📏 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    except (OSError, IOError):
        pass

def _display_failure_results():
    """Display failure results"""
    print("\n❌ MERGE FAILED!")
    print("No buildings were found or merged.")

def main():
    """Main function untuk merge tiles utility"""
    _print_header()
    
    # Parse arguments
    tiles_input, output_file = _parse_arguments()
    if tiles_input is None:
        return
    
    # Resolve tiles directory
    tiles_dir = _resolve_tiles_directory(tiles_input)
    
    # Display configuration
    print(f"📂 Input tiles directory: {tiles_dir}")
    print(f"📄 Output file: {output_file}")
    print("")
    
    # Validate directory exists
    if not _validate_tiles_directory(tiles_dir):
        return
    
    # Perform merge operation
    _perform_merge_operation(tiles_dir, output_file)

if __name__ == "__main__":
    main() 