from PointCloudUtils import PointCloudUtils


class AutoTest:
    def __init__(self):
        self.pcutils = PointCloudUtils()
        if input("Are you loading a downsampled file?: ") == "y":
            self.file_path = (
                "./Data/church_registered_ds_"
                + input("ds amnt?: ")
                + "_"
                + input("file type?: ")
            )
        else:
            print("Default file selected.")
            self.file_path = "./Data/church_registered.npy"
        self.file_ext = self.file_path[-4:]
        print("selected file:", self.file_path)
        print("file ext:", self.file_ext)

    def menu(self):
        print("Welcome to AutoTest")
        menu_selection = input(
            "\nPlease select an option from the menu:" + "\n1.) Auto Downsample" + "\nSelection: "
        )

        if menu_selection == "1":
            print("Auto Downsample Selected:")
            ds_amt_start = float(input("Downsample start value: "))
            ds_amt_end = float(input("Downsample end value: "))
            ds_amt_inc = float(input("Downsample increment value: "))
            self.pcutils.auto_downsample_data(ds_amt_start, ds_amt_end, ds_amt_inc)
            

if __name__ == "__main__":
    autotest = AutoTest()
    autotest.menu()
    