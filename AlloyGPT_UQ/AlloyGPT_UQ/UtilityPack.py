# utility block
# =========================================================
# create a folder path if not exist
import os

def create_path(this_path):
    if not os.path.exists(this_path):
        print('Creating the given path...')
        os.mkdir (this_path)
        path_stat = 1
        print('Done.')
    else:
        print('The given path already exists!')
        path_stat = 2
    return path_stat

#
def measure_model_parameters(model):
    model_size = sum(t.numel() for t in model.parameters())
    print(f"model size: {model_size/1000**2:.8f}M parameters")
    return model_size
# 
# 
def add_one_line_to_file(
    file_name=None, 
    this_line=None, 
    mode='a'
):
    #
    with open(file_name, mode) as f:
        f.write(this_line)
        
    return 0