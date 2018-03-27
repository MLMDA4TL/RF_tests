import sys

sys.path.insert()
import lib_files_s3 as lib_s3


def download_bdd_2015(path_to):
    # S3 params
    bucket_name = 'fim-algo'
    folder_name = 'LudoAndMounir/bdd_simulee_raw'
    lib_s3.download_folder(bucket_name, folder_name,
                           path_to, same_path_s3=False)


def download_bdd2015_sim_init(path_to):
    # S3 params
    bucket_name = 'fim-algo'
    folder_name = 'LudoAndMounir/BDD_ampl_simul/BDD_init_csv'
    lib_s3.download_folder(bucket_name, folder_name,
                           path_to, same_path_s3=False)


if __name__ == '__main__':
    path = "data_Tarkett/bdd_sim_2015/"
    download_bdd_2015(path)
