from PointCloudLoader import PointCloudLoader


def load():
    file_path = "./Data/church_registered_cloudCompare.las"
    pc_loader = PointCloudLoader(file_path)

    pcd = pc_loader.load_point_cloud_las(True)

def main():
    user_input = input("y/n?\n")
    if (user_input == 'y'):
        load()

if __name__=="__main__":
    main()