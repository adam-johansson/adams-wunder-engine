import pandas as pd


def load_input(input_file, input_folder):
    file_path = input_folder + input_file
    df = pd.read_csv(file_path, delimiter=" ", header=0)
    print(df["p_in"])

    return
