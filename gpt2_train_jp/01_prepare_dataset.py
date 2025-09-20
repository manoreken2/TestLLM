import os
import glob
import argparse

def combine_files(in_dir, out_dir, max_size_mb=500, separator="<|endoftext|>"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    current_content = []
    current_size = 0
    file_counter = 1

    files = glob.glob(in_dir + '/*.txt')
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        estimated_size = len(content.encode("utf-8"))

        if current_size + estimated_size > max_size_mb * 1024 * 1024:
            target_file_path = os.path.join(out_dir, f"combined_{file_counter}.txt")
            with open(target_file_path, "w", encoding="utf-8") as target_file:
                target_file.write(separator.join(current_content))
            file_counter += 1
            current_content = [content]
            current_size = estimated_size
        else:
            current_content.append(content)
            current_size += estimated_size

    if current_content:
        target_file_path = os.path.join(out_dir, f"combined_{file_counter}.txt")
        with open(target_file_path, "w", encoding="utf-8") as target_file:
            target_file.write(separator.join(current_content))
    return file_counter


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess and combine text files for pretraining")

    parser.add_argument("--in_dir", type=str, default="orig",
                        help="Directory containing the raw training files")
    parser.add_argument("--max_size_mb", type=int, default=500,
                        help="The maximum file size for each concatenated file in megabytes")
    parser.add_argument("--out_dir", type=str, default="preprocessed",
                        help="Directory where the preprocessed files will be saved")

    args = parser.parse_args()

    file_counter = combine_files(args.in_dir, args.out_dir,max_size_mb=args.max_size_mb)
    print(f"{file_counter} file(s) saved in {os.path.abspath(args.out_dir)}")
