# import suite2p
# from deeploadmat import loadmat
from determine_dprime import get_dprimes_from_dirtree as get_dprimes
import acc_path_tools as p




def get_files_with_condition_and_dprime(path, d_prime_min, function, **kwargs):
    paths = []
    all_dprimes = get_dprimes(path)
    all_files = function(path, **kwargs)
    all_files_dic = p.as_nested_dict(all_files)
    for mouse, experiment, dprime in all_dprimes:
        mouse_path = p.mouse_path(mouse, path)
        if mouse_path in all_files_dic:
            experiment_path = p.exp_path(experiment, path)
            if experiment_path in all_files_dic[mouse_path] and dprime>d_prime_min:
                paths.append(all_files_dic[mouse][experiment])
    paths = list(set(paths))
    paths.sort()
    return paths


        
if __name__ == "__main__":
    PATH = 'D:\\Local_Repository'
    main = get_files_with_condition_and_dprime
    res = main(PATH, d_prime_min = 1, function = p.get_all_files_with_name, 
               name='spks.npy')
    print(f"Identified {len(res)} satisfactory spks.npy files, at:")
    for _,file in res:
        print(file)
    res = main(PATH, d_prime_min = 1, function = p.get_all_files_with_ext, 
               ext ='.tif')
    print(f"Identified {len(res)} satisfactory tif files, at:")
    for _,file in res:
        print(file) 