# from datetime import date
from PointCloudUtils import PointCloudUtils

class Tools:
    def __init__(self):
        self.pcutils = PointCloudUtils()

    def menu(self):
        print("Welcome to Tools")
        menu_selection = input(
            "\nPlease select an option from the menu:"
            + "\n1.) Auto Downsample"
            + "\nSelection: "
        )

        if menu_selection == "1":
            print("Auto Downsample Selected:")
            ds_amt_start = float(input("Downsample start value: "))
            ds_amt_end = float(input("Downsample end value: "))
            ds_amt_inc = float(input("Downsample increment value: "))
            self.pcutils.auto_downsample_data(ds_amt_start, ds_amt_end, ds_amt_inc)


if __name__ == "__main__":
    tools = Tools()
    tools.menu()