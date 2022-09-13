from plyfile import PlyData, PlyElement
def load_ply():
    plydata = PlyData.read('./Data/church_registered_ds_0.095.ply')
    print(plydata.elements[0])
    print(plydata.elements[0].name)
    print(plydata.elements[0].data[0])
    print(plydata.elements[0].data['x'])
    
if __name__ == "__main__":
    load_ply()